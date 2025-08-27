import os
import re
import json
import string
import logging
import argparse
from tqdm import tqdm

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

from utils_train import CHAT_TEMPLATES, PROMPT_TEMPLATE, get_model_name

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler()],
)


def call_llm(model, tokenizer, prompts, max_new_tokens):
    inputs = tokenizer(prompts, padding=True, return_tensors="pt").to("cuda")
    batch_output_ids = model.generate(
        **inputs, eos_token_id=tokenizer.eos_token_id, max_new_tokens=max_new_tokens
    )
    batch_output_ids = batch_output_ids[:, inputs["input_ids"].shape[1] :]
    output_strs = tokenizer.batch_decode(batch_output_ids, skip_special_tokens=True)
    return output_strs


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def exact_match_score(prediction, ground_truth):
    return normalize_answer(prediction) == normalize_answer(ground_truth)


def f1_score(prediction, ground_truth):
    pred_tokens = normalize_answer(prediction).split()
    gold_tokens = normalize_answer(ground_truth).split()

    common = set(pred_tokens) & set(gold_tokens)
    num_same = sum(
        min(pred_tokens.count(token), gold_tokens.count(token)) for token in common
    )
    if num_same == 0:
        return 0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def calculate_metrics(preds, golds):
    em, f1 = 0, 0
    for pred, gold in zip(preds, golds):
        if isinstance(gold, list):
            _em, _f1 = 0, 0
            for g in gold:
                _em = max(exact_match_score(pred, g), _em)
                _f1 = max(f1_score(pred, g), _f1)
        else:
            _em = exact_match_score(pred, gold)
            _f1 = f1_score(pred, gold)

        em += _em
        f1 += _f1

    em = em * 100 / (len(preds) + 1e-5)
    f1 = f1 * 100 / (len(preds) + 1e-5)
    logging.info(f"EM: {em:.4f}, F1: {f1:.4f}")
    return {
        "em": em,
        "f1": f1,
    }


def evaluation(args, data, model, tokenizer):
    gold_answers, pred_answers = [], []

    for i in tqdm(range(0, len(data), args.test_batch_size)):
        batch_data = data[i : i + args.test_batch_size]

        batch_questions = [d["question"] for d in batch_data]
        batch_contexts = [d["context"] for d in batch_data]
        batch_answers = [d["answer"] for d in batch_data]

        batch_prompts = [
            PROMPT_TEMPLATE.format(context=ctx, question=ques)
            for ctx, ques in zip(batch_contexts, batch_questions)
        ]
        batch_prompts = [
            CHAT_TEMPLATES[args.model_name].format(user=prompt, assistant="")
            for prompt in batch_prompts
        ]
        batch_preds = call_llm(model, tokenizer, batch_prompts, args.max_new_tokens)

        gold_answers.extend(batch_answers)
        pred_answers.extend(batch_preds)

        for sample, answer, pred in zip(
            data[i : i + args.test_batch_size], batch_answers, batch_preds
        ):
            sample["answer"] = answer
            sample["pred"] = pred

    eval_result = calculate_metrics(pred_answers, gold_answers)

    save_folder = f"saves/{args.dataset}"
    os.makedirs(save_folder, exist_ok=True)

    save_path = (
        f"{save_folder}/{args.model_name_or_path.split('/')[-1]}_predictions.json"
    )
    with open(save_path, "w") as f:
        json.dump(data, f, indent=4)
    logging.info(f"prediction results saved to {save_path}")

    save_path = (
        f"{save_folder}/{args.model_name_or_path.split('/')[-1]}_evaluation_scores.json"
    )
    with open(save_path, "w") as f:
        json.dump(eval_result, f, ensure_ascii=False, indent=4)
    logging.info(f"evaluation results saved to {save_path}")


def run_evaluation(args, model=None, tokenizer=None):
    logging.info(f"Evaluate on {args.dataset} for the Model: {args.model_name_or_path}")
    args.model_name = get_model_name(args.model_name_or_path)

    ############################## load dataset ###############################

    args.data_path = f"data/{args.dataset}/test.json"
    with open(args.data_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    logging.info("Loaded {} instances.".format(len(data)))

    ############################## load model ###############################

    if model is None:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name_or_path, padding_side="left", trust_remote_code=True
        )
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        model = AutoModelForCausalLM.from_pretrained(
            args.model_name_or_path,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            ),
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )

        logging.info("Loaded model {}.".format(args.model_name_or_path))

        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.do_sample = False

    ############################## evaluation ###############################
    evaluation(args, data, model, tokenizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", default="nq-swap", type=str)
    parser.add_argument("--model_name_or_path", default=None, type=str)
    parser.add_argument("--test_batch_size", default=16, type=int)
    parser.add_argument("--max_new_tokens", default=32, type=int)

    args = parser.parse_args()

    run_evaluation(args)
