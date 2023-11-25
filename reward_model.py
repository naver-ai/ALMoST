import os
import torch
from torch import nn
import torch.nn.functional as F
import transformers
from transformers import LlamaForCausalLM, Trainer

import utils
import logging
from dataclasses import dataclass
from typing import Sequence, Dict
from torch.utils.data import Dataset


class RewardModelTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        # Split the inputs and rewards into two parts, chosen and rejected
        assert len(input_ids.shape) == 2
        bs = input_ids.shape[0] // 2
        
        rewards_j = model(input_ids=input_ids[:bs], attention_mask=attention_mask[:bs])
        rewards_k = model(input_ids=input_ids[bs:], attention_mask=attention_mask[bs:])
        loss = -torch.nn.functional.logsigmoid(rewards_j - rewards_k).mean()
        if return_outputs:
            return loss, {"rewards_j": rewards_j, "rewards_k": rewards_k}
        return loss


class LlamaRewardModel(LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.v_head = nn.Linear(config.hidden_size, 1, bias=False)
        self.eos_token_id = self.config.eos_token_id

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None,
        labels=None,
        return_dict=False,
        output_attentions=False,
        output_hidden_states=False,
    ):
        transformer_outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

        hidden_states = transformer_outputs[0]
        rewards = self.v_head(hidden_states).squeeze(-1)

        # Compute pairwise loss. Only backprop on the last value before padding
        sequence_lengths = (torch.ne(input_ids, self.eos_token_id).sum(-1)).to(rewards.device)
        end_scores = rewards[torch.arange(input_ids.shape[0], device=rewards.device), sequence_lengths]
        return end_scores
    

class RMDataset(Dataset):
    """Dataset for RM fine-tuning."""

    def __init__(self, data_path: str, tokenizer: transformers.PreTrainedTokenizer):
        super(RMDataset, self).__init__()
        logging.warning("Loading data...")
        list_data_dict = utils.jload(data_path)

        logging.warning("Formatting inputs...")
        
        list_data_dict = [x for x in list_data_dict if x['chosen'] != x['rejected']]

        if list_data_dict[0].get("prompt"):
            chosens = [f"{example['prompt']}\n\nAssistant: {example['chosen']}" for example in list_data_dict]
            rejecteds = [f"{example['prompt']}\n\nAssistant: {example['rejected']}" for example in list_data_dict]
        else:
            chosens = [example['chosen'] for example in list_data_dict]
            rejecteds = [example['rejected'] for example in list_data_dict]

        logging.warning("Tokenizing inputs... This may take some time...")
        data_dict = tokenizer(
                [line + f"{tokenizer.eos_token}" for line in chosens + rejecteds],
                truncation=True,
                max_length=tokenizer.model_max_length,
                padding="longest",
                add_special_tokens=False,
                return_tensors="pt",
            )
        
        chosen_input_ids = data_dict["input_ids"][:len(chosens)]
        reject_input_ids = data_dict["input_ids"][len(chosens):]

        self.input_ids = [[c, j] for c, j in zip(chosen_input_ids, reject_input_ids)]
        self.labels =  torch.tensor([0] * len(list_data_dict) + [1] * len(list_data_dict))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        return dict(input_ids=self.input_ids[i],
                    labels=self.labels[i])


@dataclass
class DataCollatorForRMDataset(object):
    """Collate examples for RM fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances] for key in ("input_ids", "labels"))
        
        chosen_input_ids, reject_input_ids = list(zip(*input_ids))
        chosen_input_ids = torch.stack(chosen_input_ids)
        reject_input_ids = torch.stack(reject_input_ids)
        
        input_ids = torch.cat((chosen_input_ids, reject_input_ids))
        return dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )


def make_rm_data_module(tokenizer: transformers.PreTrainedTokenizer, data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = RMDataset(tokenizer=tokenizer, data_path=data_args.data_path)
    data_collator = DataCollatorForRMDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset, eval_dataset=None, data_collator=data_collator)
