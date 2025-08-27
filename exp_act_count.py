import os
import json
import random
import logging
import argparse
import numpy as np

import torch
from tqdm import tqdm
from transformers import AutoTokenizer, BitsAndBytesConfig

from utils_train import (
    PROMPT_TEMPLATE,
    CHAT_TEMPLATES,
    MODEL_INFO,
    get_model_name,
    set_seed,
)


def expert_act_count(all_router_weights, num_act_experts):
    num_samples = len(all_router_weights)
    num_layers = all_router_weights[0].shape[0]
    num_experts = all_router_weights[0].shape[-1]
    binary_activation = np.zeros([num_samples, num_layers, num_experts], dtype=float)
    for i in tqdm(range(num_samples)):
        router_weights = all_router_weights[i]
        seq_len = router_weights.shape[1]
        for j in range(num_layers):
            for k in range(seq_len):
                unit_sample_weights = router_weights[j, k, :]
                act_experts = np.argsort(unit_sample_weights)[-num_act_experts:]
                binary_activation[i, j, act_experts] += 1
    activation_counts = np.sum(binary_activation, axis=0)
    return activation_counts


def get_router_weights(
    model, dataset, model_name, tokenizer, max_sample=-1, max_seq_length=-1
):
    ############################## load dataset ###############################
    with open(f"data/{dataset}/train.json", "r") as fh:
        data = json.load(fh)
    logging.info(f"Dataset {dataset} loaded.")
    random.shuffle(data)

    ############################## hooking ###############################
    all_router_weights = []
    model.eval()
    with torch.no_grad():
        for idx in tqdm(
            range(min(max_sample, len(data)) if max_sample > 0 else len(data))
        ):
            sample = data[idx]
            question = sample["question"]
            context = sample["context"]
            answer = sample["answer"]
            prompt = PROMPT_TEMPLATE.format(context=context, question=question)
            prompt = CHAT_TEMPLATES[model_name].format(user=prompt, assistant=answer)

            inputs_pt = tokenizer(
                [prompt], padding=True, return_tensors="pt", add_special_tokens=False
            ).to("cuda")
            if (
                max_seq_length != -1
                and inputs_pt["input_ids"].shape[1] > max_seq_length
            ):
                continue
            outputs = model(
                **inputs_pt,
                output_hidden_states=False,
                output_attentions=False,
                output_router_logits=True,
                return_dict=True,
            )
            router_weights = torch.cat(
                [rw.cpu().detach() for rw in outputs["router_logits"]], 0
            )

            all_router_weights += [router_weights]

    return all_router_weights


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--max_sample", type=int, default=-1)
    parser.add_argument("--max_seq_length", type=int, default=-1)
    parser.add_argument("--model_name_or_path", type=str)

    args = parser.parse_args()
    print(args, flush=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    set_seed(args.seed)

    ############################## load model ###############################
    args.model_name = get_model_name(args.model_name_or_path)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, padding_side="left", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    MODEL_CLASS = MODEL_INFO[args.model_name]["class"]

    model = MODEL_CLASS.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id
    model.generation_config.do_sample = False

    logging.info(f"Model {args.model_name_or_path} loaded.")

    all_router_weights = get_router_weights(
        model,
        args.dataset,
        args.model_name,
        tokenizer,
        args.max_sample,
        args.max_seq_length,
    )

    logging.info(f"Got {len(all_router_weights)} router weights for {args.dataset}.")

    num_act_experts = MODEL_INFO[args.model_name]["num_act_experts"]
    expert_act_matrix = expert_act_count(all_router_weights, num_act_experts)

    save_path = f"saves/{args.dataset}/{os.path.basename(args.model_name_or_path)}_expert_act_matrix.pth"
    torch.save(expert_act_matrix, save_path)
    logging.info(f"Saved router weights to {save_path}.")


if __name__ == "__main__":
    main()

