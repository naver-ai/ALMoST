import os
import json
import torch
import argparse
import subprocess
import numpy as np
from datasets import load_dataset
from collections import defaultdict
from conversation import get_conv_template
from eval_prompt import static_HHH_prompt, truthfulQA_prompt
from transformers import AutoModelForCausalLM, AutoTokenizer


def run_vicuna_eval(
    model,
    tokenizer,
    data,
    model_name,
    output_dir,
    baseline_model_name=None,
    top_p=1.0,
    temperature=0.7,
    max_new_tokens=1024
):
    
    def construct_single_turn_prompt(conv_template, msg):
        conv_template.clear()
        conv_template.append_message(conv_template.roles[0], msg)
        conv_template.append_message(conv_template.roles[-1], "")
        return conv_template.get_prompt()
    
    if "alpaca" in model_name:
        conv_template = get_conv_template("alpaca")
    elif "vicuna" in model_name:
        conv_template = get_conv_template("vicuna_v1.1")
    else:
        conv_template = get_conv_template("almost")
    
    if not os.path.exists(f"{output_dir}/{model_name}.jsonl"):
        responses = []
        for idx, question in enumerate(data):
            prompt = construct_single_turn_prompt(conv_template, question['text'])
            inputs = tokenizer(prompt, return_tensors="pt")
            output = model.generate(input_ids=inputs.input_ids.to(model.device),
                                    attention_mask=inputs.attention_mask.to(model.device),
                                    do_sample=True,
                                    use_cache=True,
                                    top_p=top_p,
                                    temperature=temperature,
                                    pad_token_id=tokenizer.pad_token_id,
                                    max_new_tokens=max_new_tokens)

            if "dolly" in model_name or "oasst" in model_name:
                text_output = tokenizer.decode(output[0])
                text_output = text_output.replace(prompt, "").split("### End")[0]
                text_output = text_output.replace(prompt, "").split(tokenizer.eos_token)[0]
            else:
                text_output = tokenizer.decode(output[0], skip_special_tokens=True).replace(prompt, "")
            responses.append(text_output)
            print(f"[Vicuna-eval] {model_name}: {idx}/{len(data)}", end="\r", flush=True)

        with open(f"{output_dir}/{model_name}.jsonl", "w", encoding="utf-8") as f:
            for i, text in enumerate(responses):
                    dic = {
                        'question_id': i + 1,
                        'text': text,
                        'answer_id': f'{model_name}-{i+1}',
                        'model_id': model_name,
                        'metadata': {}
                    }
                    f.write(json.dumps(dic) + "\n")

                    
    if baseline_model_name:
        commands = [
          f"""python eval_gpt_review.py \
          -q data/vicuna/question.jsonl \
          -a {output_dir}/{baseline_model_name}.jsonl {output_dir}/{model_name}.jsonl \
          -p data/vicuna/prompt.jsonl \
          -r data/vicuna/reviewer.jsonl \
          -o {output_dir}/vicuna_{baseline_model_name}_vs_{model_name}.jsonl \
          --bidirectional"""
        ]

        command = '; '.join(line.strip() for line in commands)
        p = subprocess.Popen(command, shell=True)
        p.wait()

        reviews = []
        for line in open(f"{output_dir}/vicuna_{baseline_model_name}_vs_{model_name}.jsonl"):
            line = json.loads(line)
            reviews.append(line)

        win, loss, tie = 0, 0, 0
        our_scores = []
        baseline_scores = []
        for i, r in enumerate(reviews):
            ours_idx = 1 if i < len(data) else 0
            another_idx = 0 if i < len(data) else 1
            if r['score'][ours_idx] > r['score'][another_idx]:
                win += 1
            elif r['score'][ours_idx] == r['score'][another_idx]:
                tie += 1
            elif r['score'][ours_idx] < r['score'][another_idx]:
                loss += 1

            our_scores.append(r['score'][ours_idx])
            baseline_scores.append(r['score'][another_idx])

        result = {
            f"{model_name}_win": win,
            "tie": tie,
            f"{baseline_model_name}_win": loss,
            f"{model_name}_scores": sum(our_scores),
            f"{baseline_model_name}_scores": sum(baseline_scores)
        }
        print(json.dumps(result, indent=2, ensure_ascii=False))
        json.dump(result, open(f"{output_dir}/vicuna_{baseline_model_name}_vs_{model_name}.json", "w", encoding="utf-8"))


def run_static_hhh_eval(
    model,
    tokenizer,
    data,
    model_name,
    output_dir
):
    A_idx, B_idx = tokenizer.encode("A")[-1], tokenizer.encode("B")[-1]
    hit, cnt = 0, 0
    pred_dict = defaultdict(list)
    preds = []
    pred_pairs = []
    for idx in range(len(data)):
        inputs = []
        query, answer_a = data[idx]['chosen'].rsplit("\n\nAssistant:", 1)
        query, answer_b = data[idx]['rejected'].rsplit("\n\nAssistant:", 1)
        query = query.replace("Human:", "Query:").replace("Assistant:", "Response:")
        
        # Bi-positional prediction
        ab = static_HHH_prompt.format_map(
            {'question': query,
             'answer_a': answer_a.strip(),
             'answer_b': answer_b.strip()
            }
        )
        ba = static_HHH_prompt.format_map(
            {'question': query,
             'answer_a': answer_b.strip(),
             'answer_b': answer_a.strip()
            }
        )
        inputs.append(ab)
        inputs.append(ba)
        inputs = tokenizer(inputs,
                           padding='longest',
                           truncation=True,
                           max_length=1024,
                           return_tensors="pt")
        output = model.generate(input_ids=inputs.input_ids.to(model.device),
                                attention_mask=inputs.attention_mask.to(model.device),
                                do_sample=False,
                                pad_token_id=tokenizer.pad_token_id,
                                temperature=0.,
                                top_k=1,
                                max_new_tokens=1,
                                return_dict_in_generate=True,
                                output_scores=True)
        preda_1 = torch.softmax(output.scores[0], -1)[0][A_idx]
        predb_1 = torch.softmax(output.scores[0], -1)[0][B_idx]

        preda_2 = torch.softmax(output.scores[0], -1)[1][A_idx]
        predb_2 = torch.softmax(output.scores[0], -1)[1][B_idx]

        preda = preda_1 + predb_2
        predb = predb_1 + preda_2

        pred_pairs.append([preda.item(), predb.item()])

        if preda >= predb: 
            hit += 1
            pred_dict[data[idx].get('type', 'all')].append(True)
            preds.append(True)
        else:
            pred_dict[data[idx].get('type', 'all')].append(False)
            preds.append(False)

        cnt += 1

        print(f"[HHH-eval] {model_name}: {idx}/{len(data)} {hit / cnt}", end="\r", flush=True)
    result = {
        'helpful': np.mean(pred_dict['helpful']),
        'harmless': np.mean(pred_dict['harmless']),
        'honest': np.mean(pred_dict['honest']),
        'other': np.mean(pred_dict['other']),
        'all': np.mean(preds)
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    json.dump(result, open(f"{output_dir}/static_hhh_{model_name}.json", "w", encoding="utf-8"))
    
    
def run_truthfulqa_eval(
    model,
    tokenizer,
    data,
    model_name,
    output_dir
):
    target_idx = tokenizer.encode("true")[-1]

    hit, cnt = 0, 0
    preds = []
    for idx in range(len(data['validation'])):
        question = data['validation']['question'][idx]
        scores = []
        for k in data['validation']['mc1_targets'][idx]['choices']:
            dic = dict(question=question, answer=k)
            inputs = [truthfulQA_prompt.format_map(dic)]
            inputs = tokenizer(inputs,
                               padding='longest',
                               truncation=True,
                               max_length=512,
                               return_tensors="pt")
            output = model.generate(input_ids=inputs.input_ids.to(model.device),
                                    attention_mask=inputs.attention_mask.to(model.device),
                                    do_sample=False,
                                    pad_token_id=tokenizer.pad_token_id,
                                    top_k=1,
                                    max_new_tokens=1,
                                    return_dict_in_generate=True,
                                    output_scores=True)
            scores.append(torch.softmax(output.scores[0], -1)[:,target_idx].item())
        pred = np.array(scores).argmax().item()
        if pred == 0:
            hit += 1

        cnt += 1
        print(f"[TruthfulQA] {model_name}: {idx}/{len(data['validation'])} {hit / cnt}", end="\r", flush=True)
    
    result = {
        'MC1-Acc': hit / cnt
    }
    print(json.dumps(result, indent=2, ensure_ascii=False))
    json.dump(result, open(f"{output_dir}/truthfulqa_{model_name}.json", "w", encoding="utf-8"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--output_dir',
                        type=str,
                        default='outputs')
    parser.add_argument('--benchmark_name',
                        type=str,
                        help="specific benchmark to evaluate e.g., (hhh|truthful|vicuna)"
                       )
    parser.add_argument('--baseline_model_name',
                        type=str,
                        help="baseline model to compare for vicuna-bench"
                        "{output_dir}/{baseline_model_name}.jsonl file is required" 
                       )
    args = parser.parse_args()
    
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        device_map={"": "cuda" if torch.cuda.is_available() else "cpu"},
        torch_dtype=torch.bfloat16,
    )
    model.eval()
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    model_name = args.model_name_or_path.split("/")[-1].strip("/")
    
    if not args.benchmark_name or args.benchmark_name == "hhh":
        data = json.load(open("data/hhh_eval.json"))
        run_static_hhh_eval(model, tokenizer, data, model_name, args.output_dir)
        
    if not args.benchmark_name or args.benchmark_name == "truthful":
        data = load_dataset("truthful_qa", 'multiple_choice')
        run_truthfulqa_eval(model, tokenizer, data, model_name, args.output_dir)
        
    if not args.benchmark_name or args.benchmark_name == "vicuna":
        data = [json.loads(line) for line in open("data/vicuna/question.jsonl")]
        run_vicuna_eval(model, tokenizer, data, model_name, args.output_dir, args.baseline_model_name)
    
    
    