"""
ALMoST
Copyright (c) 2023-present NAVER Cloud Corp.
Apache-2.0
"""

import os
import re
import json
import argparse
import itertools
import numpy as np
from tqdm import tqdm
from utils import get_logger
from typing import List, Dict, Optional
from collections import Counter
from generator import PromptGenerator, PromptedResponseGenerator, SamplingConfig


logger = get_logger(__name__)


class SyntheticRanker:
    
    def __init__(self, rubric: List[str]):
        self.rubric = rubric
        self.set_rank()
        
    def load_dataset(self, target_dir: str):
        def flatten_list(lst):
            flatten = []
            for l in lst:
                if isinstance(l, list):
                    flatten.extend(l)
                else:
                    flatten.append(l)
            return flatten
        
        prompts = open(f"{target_dir}/prompts.txt", "r", encoding="utf-8").read().splitlines()
        config_names = []
        all_responses = []
        for i, config_name in enumerate(flatten_list(self.rubric)):
            responses = json.load(open(f"{target_dir}/{config_name}.json", "r", encoding="utf-8"))
            assert len(prompts) == len(responses)

            all_responses.append(responses)
            config_names.append(config_name)
        
        all_responses = list(zip(*all_responses))
        instances = []
        for idx in range(len(prompts)):
            p = prompts[idx]
            dic = {
                'prompt': p
            }
            for i, r in enumerate(all_responses[idx]):
                dic[config_names[i]] = r
            instances.append(dic)
        return instances
        
    def set_rank(self):
        synthetic_ranking = {}
        rank = 0
        for r in self.rubric:
            if isinstance(r, list):
                for rr in r:
                    synthetic_ranking[rr] = rank
            else:
                synthetic_ranking[r] = rank
            rank += 1
        self.synthetic_ranking = synthetic_ranking

    def get_comparison(self, instance: dict) -> List[dict]:
        """
        Each instance is dictionary
        It includes input 'prompt' and
        candidate responses (value) with corresponding model_name as a key.
        """
        
        prompt = None
        candidates = []
        responses = []
        ranking = []
        for k, v in instance.items():
            if k == "prompt":
                prompt = v
                continue

            if not v.strip() or v == "BAD":
                continue

            candidates.append(k)
            responses.append(v)
            ranking.append(self.synthetic_ranking.get(k))
        
        gathered = list(zip(candidates, responses, ranking))
        gathered = sorted(gathered, key=lambda x: x[-1])
        
        if len(gathered) < 2:
            return []
        
        length_variance = self.check_length(instance)
        
        ranking_instances = []
        checked = []
        rank_checked = []
        for combi in itertools.combinations(gathered, 2):
            if combi[0][0] in checked:
                continue

            if combi[0][-1] in rank_checked:
                continue

            if len(combi[0][1]) < len(combi[1][1]) and not length_variance[combi[0][0]]:
                continue

            if combi[0][-1] == combi[1][-1]:
                continue

            dic = {
                "prompt": prompt,
                "chosen": combi[0][1],
                "rejected": combi[1][1],
                "meta": f"{combi[0][0]}-vs-{combi[1][0]}"
            }
            checked.append(combi[0][0])
            rank_checked.append(combi[0][-1])
            ranking_instances.append(dic)
        return ranking_instances

    def check_length(self, instance: dict) -> dict:
        lengths = []
        for k, v in instance.items():
            if k == "prompt":
                continue
            
            if v == "BAD":
                continue

            lengths.append(len(v))

        avg_length = np.mean(lengths)
        std_length = np.std(lengths)

        length_variance = {}
        for k, v in instance.items():
            if k == "prompt":
                continue
                
            length_variance[k] = True if len(v) >= (avg_length - std_length / 2) else False
        return length_variance

    
def prompt_generation(
    model_name_or_path: str,
    prompt_file_path: str,
    output_dir: str,
    num_generation: int = 100,
    batch_size: int = 4,
    n_gpu: int = 1,
    use_vllm: bool = True,
    cache_dir: str = None
) -> None:
    config = SamplingConfig(
        prompt_file_path=prompt_file_path,
        model_name_or_path=model_name_or_path,
        n_shot=10,
        batch_size=batch_size,
        temperature=1.2,
        top_p=0.9,
        max_new_tokens=64,
        prompt_splitter="\n",
        static_prompt=False,
        use_vllm=True,
        n_gpu=n_gpu,
        cache_dir=cache_dir
    )
    generator = PromptGenerator(config)
    generator.load_model()
    prompts = generator.generate(num_generation)
    with open(f"{output_dir}/prompts.txt", "w", encoding="utf-8") as f:
        for prompt in prompts:
            f.write(prompt + "\n")
    
    
def response_generation(
    prompts: List[str],
    model_name_or_path: str,
    prompt_file_path: str,
    n_shot: int,
    output_dir: str,
    batch_size: int =4,
    n_gpu: int = 1,
    use_vllm: bool = True,
    cache_dir: str = None
) -> None:

    config = SamplingConfig(
        prompt_file_path=prompt_file_path,
        model_name_or_path=model_name_or_path,
        n_shot=n_shot,
        batch_size=batch_size,
        temperature=1.0,
        top_p=0.9,
        max_new_tokens=768 if "Faithful" in prompt_file_path else 384,
        prompt_splitter="\n\n-----",
        static_prompt=True,
        use_vllm=True,
        n_gpu=n_gpu,
        cache_dir=cache_dir
    )
    config_name = model_name_or_path.split("/")[-1].lower() + \
    "-" + prompt_file_path.split("/")[-1].replace("_prompt.txt", "") + \
    f"-{n_shot}shot"
    
    generator = PromptedResponseGenerator(config)
    generator.load_model()
    responses = generator.generate(prompts)
    with open(f"{output_dir}/{config_name}.json", "w", encoding="utf-8") as f:
        json.dump(responses, f)
        
        
def construct_synthetic_comparison(
    output_dir: str,
    rubric: List[str]
):
    ranker = SyntheticRanker(rubric)
    instances = ranker.load_dataset(output_dir)

    comparison_dataset = []
    for instance in instances:
        comparison = ranker.get_comparison(instance)
        comparison_dataset.extend(comparison)
        
    with open(f"{output_dir}/comparison.json", "w", encoding="utf-8") as f:
        json.dump(comparison_dataset, f)


def main(args):
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode in ["prompt_generation", "pg"]:
        prompt_generation(
            args.model_name_or_path,
            args.prompt_file_path,
            args.output_dir,
            args.num_generation,
            args.batch_size,
            args.n_gpu,
            args.use_vllm,
            args.cache_dir
        )
        
    if args.mode in ["response_generation", "rg"]:
        
        with open(f"{args.output_dir}/prompts.txt", "r", encoding="utf-8") as f:
            prompts = f.read().splitlines()

        response_generation(
            prompts,
            args.model_name_or_path,
            args.prompt_file_path,
            args.n_shot,
            args.output_dir,
            args.batch_size,
            args.n_gpu,
            args.use_vllm,
            args.cache_dir
        )
        
    if args.mode == "cs":
        construct_synthetic_comparison(
            args.output_dir,
            args.rubric
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', type=str)
    parser.add_argument("--mode",
                        help="The mode should be one of ['pg', 'rg', 'cs'].",
                        type=str,
                        required=True)
    parser.add_argument('--model_name_or_path', type=str)
    parser.add_argument('--prompt_file_path', type=str)
    parser.add_argument('--n_shot', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_generation', type=int, default=100)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--rubric', nargs="+", default=[])
    parser.add_argument('--cache_dir', type=str)
    parser.add_argument('--use_hf', action="store_true", default=False)
    args = parser.parse_args()
    args.use_vllm = False if args.use_hf else True
    assert args.mode in ['pg', 'rg', 'cs']
    main(args)