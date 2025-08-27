import os
import random
import numpy as np

import torch
from torch import nn
from trl import SFTTrainer

from models.modeling_olmoe import OlmoeForCausalLM


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


def get_model_name(model_name_or_path):
    for model_name in CHAT_TEMPLATES:
        if model_name in model_name_or_path:
            return model_name


MODEL_INFO = {
    "OLMoE-1B-7B-0924-Instruct": {
        "class": OlmoeForCausalLM,
        "num_layers": 16,
        "num_experts": 64,
        "num_act_experts": 8,
    },
}


CHAT_TEMPLATES = {
    "OLMoE-1B-7B-0924-Instruct": "<|endoftext|><|user|>\n{user}\n<|assistant|>\n{assistant}",
}


PROMPT_TEMPLATE = """Based on the context below, output the correct answer for the following question.

Context:
{context}

Question:
{question}"""


def get_target_module(args):
    if args.training_method.startswith("ceft"):
        num_experts = int(args.training_method.split("-")[1])
        exp_act_matrix_file = args.expert_act_matrix_path
        exp_act_matrix = torch.load(exp_act_matrix_file, weights_only=False)
        sorted_indices = np.argsort(-exp_act_matrix, axis=1)
        all_exp_ids = sorted_indices[:, :num_experts].tolist()

        target_modules = []
        for i in range(len(all_exp_ids)):
            for j in all_exp_ids[i]:
                if args.model_name == "OLMoE-1B-7B-0924-Instruct":
                    target_modules += [
                        f"layers.{i}.mlp.experts.{j}.gate_proj",
                        f"layers.{i}.mlp.experts.{j}.up_proj",
                        f"layers.{i}.mlp.experts.{j}.down_proj",
                    ]

    elif args.training_method.startswith("rt"):
        if args.model_name == "OLMoE-1B-7B-0924-Instruct":
            target_modules = ["mlp.gate"]

    return target_modules


class CustomTrainer(SFTTrainer):
    def __init__(self, end_of_prompt_ids, eos_token_id, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.eop_ids = end_of_prompt_ids
        self.eos_token_id = eos_token_id

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        labels = inputs["labels"]

        len_eop = len(self.eop_ids)
        for i in range(len(labels)):
            label = labels[i].cpu().numpy().tolist()
            for j in range(len(label), -1, -1):
                if label[j - len_eop : j] == self.eop_ids:
                    break
            if j == 0:
                j = len(label)
            labels[i, :j] = -100

        labels[:, -1] = self.eos_token_id

        # forward pass
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Shift so that tokens < n predict n
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = nn.CrossEntropyLoss()
        shift_logits = shift_logits.view(-1, model.config.vocab_size)
        shift_labels = shift_labels.view(-1)
        # Enable model parallelism
        shift_labels = shift_labels.to(shift_logits.device)
        loss = loss_fct(shift_logits, shift_labels)

        return (loss, outputs) if return_outputs else loss
