"""
ALMoST
Copyright (c) 2023-present NAVER Cloud Corp.
Apache-2.0
"""

import os
import re
import torch
import random
import string
import dataclasses
from typing import List
from tqdm import tqdm
from rouge import Rouge
from utils import get_logger, TextBatchLoader
from vllm import LLM, SamplingParams
from transformers import AutoModelForCausalLM, AutoTokenizer, BertTokenizerFast


os.environ["TOKENIZERS_PARALLELISM"] = "false"
logger = get_logger(__name__)


@dataclasses.dataclass
class SamplingConfig:
    prompt_file_path: str
    model_name_or_path: str
    n_shot: int
    batch_size: int
    temperature: float
    top_p: float
    max_new_tokens: int
    prompt_splitter: str
    static_prompt: bool = True
    use_vllm: bool = True
    n_gpu: int = 1
    cache_dir: str = None
        

class Generator:

    def __init__(self, config: SamplingConfig):
        self.config = config
        
    def load_seed_prompt(self):
        demon = open(self.config.prompt_file_path, "r", encoding="utf-8").read()
        demon = [d for d in demon.split(self.config.prompt_splitter) if d.strip("\n ")]
        
        if not self.config.static_prompt:
            # Note the first element should be instruction of generation task.
            return demon[0], demon[1:]

        if self.config.n_shot:
            demon = demon[:self.config.n_shot + 1]

        return self.config.prompt_splitter.join(demon) + self.config.prompt_splitter
    
    def load_model(self):
        if self.config.use_vllm:
            self.init_vllm_model(self.config.model_name_or_path)
        else:
            self.init_hf_model(self.config.model_name_or_path)

    def init_vllm_model(self, model_name_or_path):
        logger.info(f"Loading {model_name_or_path} via vLLM...")
        self.model = LLM(model=model_name_or_path,
                         download_dir=self.config.cache_dir,
                         tensor_parallel_size=self.config.n_gpu)
        logger.info("Model is prepared!")
        
    def init_hf_model(self, model_name_or_path, device=0):
        logger.info(f"Loading {model_name_or_path} to cuda:{device}...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            cache_dir=self.config.cache_dir
        )
        self.tokenizer.padding_side = "left"

        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            device_map={"": device},
            torch_dtype=torch.bfloat16,
            cache_dir=self.config.cache_dir
        )
        logger.info("Model is prepared!")

    def generate(self):
        raise NotImplementedError
        
    def _generate(self,
                  batch_texts: List[str],
                  temperature: float,
                  top_p: float,
                  max_new_tokens: int
                 ) -> List[str]:
        if self.config.use_vllm:
            sampling_params = SamplingParams(
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_new_tokens
            )
            outputs = self.model.generate(batch_texts, sampling_params)
            outputs = [o.outputs[0].text for o in outputs]
        else:
            batch = self.tokenizer(
                batch_texts,
                padding="longest",
                truncation=True,
                return_tensors="pt", 
                add_special_tokens=False
            )
            generated = self.model.generate(
                batch["input_ids"].to(self.model.device),
                do_sample=True,
                use_cache=True,
                temperature=temperature,
                top_p=top_p,
                max_new_tokens=max_new_tokens
            )
            outputs = self.tokenizer.batch_decode(generated, skip_special_tokens=True)
            
        results = []
        for text, output in zip(batch_texts, outputs):
            output = output.replace(text, "")
            results.append(output)   
        return results
    
    
class PromptGenerator(Generator):

    def __init__(self, config: SamplingConfig):
        self.config = config
        inst, prompts = self.load_seed_prompt()
        self.instruction = inst
        self.seed_prompts = prompts
        self.rouge_tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
        self.prepare_seed_tokens()
        self.rouge = Rouge()
        self.filtered = []

    def generate(self,
                 target_num,
                 prefix="Query:",
                 threshold=0.5,
                 min_len=30,
                 max_len=250):
        """
        target_num: The number of prompts to generate
        prefix: prefix to be prepended to the few-shot prompts
        threshold: Rouge score to check token overlap with already generated prompts
        """
        
        tbar = tqdm(
            total=target_num,
            desc=f"Prompt Generation",
            dynamic_ncols=True
        )
        mined_prompts = []
        cnt = 0
        while len(mined_prompts) < target_num:
            try:
                batch_prompt = []
                for _ in range(self.config.batch_size):
                    p = self.form_dynamic_prompt(
                        mined_prompts,
                        prefix,
                        self.config.n_shot
                    )
                    batch_prompt.append(p)
                 
                outputs = self._generate(
                    batch_prompt,
                    self.config.temperature,
                    self.config.top_p,
                    self.config.max_new_tokens
                )

                results = []
                for output in outputs:
                    extracted = ""
                    for line in output.split("\n"):
                        if line.strip().startswith(prefix):
                            extracted = line.replace(prefix, "").strip()
                            extracted = re.sub(
                                r'query\s?[:]',
                                '[SEP]',
                                extracted,
                                flags=re.IGNORECASE
                            )
                            extracted = extracted.split('[SEP]')[0].strip("\n ")
                            break
                            
                    if not extracted.strip():
                        continue

                    if self.check_bad(extracted, min_len, max_len):
                        self.filtered.append(extracted)
                        continue

                    if self.check_overlap(extracted, threshold):
                        self.filtered.append(extracted)
                        continue

                    self.add_tokens(extracted)
                    mined_prompts.append(extracted)
                    results.append(extracted)

                tbar.update(len(results))
                tbar.set_postfix({
                    "Progress": len(mined_prompts) / target_num,
                })
            except KeyboardInterrupt:
                break

        return mined_prompts

    def form_dynamic_prompt(self, mined_prompts, prefix="Query:", n_shot=3):
        shots = random.sample(self.seed_prompts + mined_prompts, n_shot)
        p = f"{self.instruction}\n\n"
        for s in shots:
            p += f"{prefix} {s}\n"
        p += prefix
        return p

    def add_tokens(self, prompt):
        self.tokens.append(" ".join(self.rouge_tokenizer.tokenize(prompt)))

    def prepare_seed_tokens(self):
        self.tokens = []
        for p in self.seed_prompts:
            self.add_tokens(p)

    def check_overlap(self, prompt, threshold=0.7):
        max_score = 0.
        q = " ".join(self.rouge_tokenizer.tokenize(prompt))
        for c in self.tokens:
            score = self.rouge.get_scores(q, c, avg=True)['rouge-l']['f']

            if score > max_score:
                max_score = score

        if max_score >= threshold:
            return True
        return False

    def check_bad(self, prompt, min_len, max_len, prefix=None):
        if prefix and prefix in prompt:
            return True
        
        if len(prompt) < min_len and prompt[-1] not in string.punctuation:
            return True
        
        if len(prompt) > max_len and not prompt[-1] not in string.punctuation:
            return True
        
        return False
    
    
class PromptedResponseGenerator(Generator):

    def __init__(self, config):
        self.config = config
        self.static_prompt = self.load_seed_prompt()

    def form_prompt(self, prompt):
        return self.static_prompt + f"\n\nHuman: {prompt}\n\nAssistant: "
    
    def generate(self, prompts, turn=0, min_length=50):
        tbar = tqdm(
            total=len(prompts),
            desc=f"Response Generation",
            dynamic_ncols=True
        )
        mined_responses = []
        cnt = 0
        data_loader = TextBatchLoader(prompts, self.config.batch_size)
        
        for idx, (indices, batch) in enumerate(data_loader):
            try:
                batch = [self.form_prompt(p) for p in batch]

                outputs = self._generate(
                        batch,
                        self.config.temperature,
                        self.config.top_p,
                        self.config.max_new_tokens
                )

                for output in outputs:
                    try:
                        output = output.split(self.config.prompt_splitter)[0]
                        output = output.split("Assistant:")[turn]
                        output = output.split("Human:")[0].strip("\n ")
                    except:
                        output = "BAD"
                    
                    if self.check_bad(output, min_length):
                        output = "BAD"
                    
                    mined_responses.append(output)
                tbar.update(len(outputs))
                tbar.set_postfix({
                    "Progress": len(mined_responses) / len(prompts),
                })

            except KeyboardInterrupt:
                break
                
        return mined_responses
    
    def check_bad(self, response, min_length):
        if response == "BAD":
            return True
        
        if len(response) < min_length:
            return True

        if response.endswith("?") or response.endswith(":"):
            return True
        
        if response.lower().startswith("well"):
            return True
        
        if response[-1] not in [".", "!"]:
            return True
        
        if re.search(r"(don('|’)t|do not)\s+know", response, flags=re.IGNORECASE):
            return True
        
        if re.search(r"Human[:]", response, flags=re.IGNORECASE):
            return True

        if re.search(r"[<]Image[>]", response, flags=re.IGNORECASE):
            return True

        return False
