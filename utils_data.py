import json
from tqdm import tqdm
from datasets import Dataset
from utils_train import CHAT_TEMPLATES, PROMPT_TEMPLATE


def load_data(args):
    with open(f"data/{args.dataset}/train.json", "r") as fh:
        train_data = json.load(fh)
    with open(f"data/{args.dataset}/dev.json", "r") as fh:
        val_data = json.load(fh)

    return train_data, val_data


def build_dataset(data, args, eos_token):
    prompts = []
    completions = []

    for idx in tqdm(range(len(data))):
        sample = data[idx]
        question = sample["question"]
        context = sample["context"]
        correct_answer = sample["answer"]
        prompt = PROMPT_TEMPLATE.format(context=context, question=question)
        prompt = CHAT_TEMPLATES[args.model_name].format(user=prompt, assistant="")
        completion = correct_answer + eos_token

        if len(args.tokenizer.tokenize(prompt + completion)) > (args.max_seq_length):
            continue

        prompts += [prompt]
        completions += [completion]

    dataset = Dataset.from_dict({"prompt": prompts, "completion": completions})
    return dataset
