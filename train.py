import os
import shutil
import logging
import argparse

import torch
from trl import SFTConfig
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from evaluation import run_evaluation
from utils_data import load_data, build_dataset
from utils_train import CustomTrainer, set_seed, get_target_module, get_model_name

os.environ["WANDB_DISABLED"] = "true"


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--max_seq_length", type=int)

    parser.add_argument("--model_name_or_path", type=str)
    parser.add_argument("--expert_act_matrix_path", type=str, default=None)
    parser.add_argument("--training_method", type=str)

    parser.add_argument("--epoch", type=float)
    parser.add_argument("--learning_rate", type=float)
    parser.add_argument("--per_device_train_batch_size", type=int)
    parser.add_argument("--gradient_accumulation_steps", type=int)
    parser.add_argument("--test_batch_size", type=int)
    parser.add_argument("--max_new_tokens", default=32, type=int)
    parser.add_argument("--save_path", type=str)

    args = parser.parse_args()
    print(args, flush=True)

    set_seed(args.seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler()],
    )

    ############################## load model ###############################
    args.model_name = get_model_name(args.model_name_or_path)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path, padding_side="left", trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    args.tokenizer = tokenizer

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        llm_int8_threshold=6.0,
        llm_int8_has_fp16_weight=False,
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=quant_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    target_modules = get_target_module(args)
    peft_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
    )

    logging.info("Loaded {}.".format(args.model_name_or_path))
    logging.info("Training {} modules.".format(target_modules))

    ############################## load dataset ###############################

    train_data, val_data = load_data(args)
    train_dataset = build_dataset(train_data, args, tokenizer.eos_token)
    val_dataset = build_dataset(val_data, args, tokenizer.eos_token)

    logging.info("Build {} training samples.".format(train_dataset.__len__()))
    logging.info("Build {} val samples.".format(val_dataset.__len__()))

    ############################## training ##############################
    run_name = f"{args.model_name}_{args.training_method}_e{int(args.epoch)}_lr{args.learning_rate}_sd{args.seed}"

    tmp_path = f"tmp_{run_name}"
    training_args = SFTConfig(
        report_to=None,
        run_name=run_name,
        logging_steps=1,
        max_seq_length=args.max_seq_length,
        output_dir=tmp_path,
        overwrite_output_dir=True,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        num_train_epochs=args.epoch,
        save_steps=0.1,
        eval_steps=0.1,
        eval_on_start=False,
        seed=args.seed,
        data_seed=args.seed,
        load_best_model_at_end=True,
        evaluation_strategy="steps",
    )

    end_of_prompt_ids = {
        "OLMoE-1B-7B-0924-Instruct": tokenizer(
            "<|assistant|>\n", add_special_tokens=False
        )["input_ids"],
    }[args.model_name]

    trainer = CustomTrainer(
        end_of_prompt_ids=end_of_prompt_ids,
        eos_token_id=tokenizer.eos_token_id,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        processing_class=tokenizer,
        args=training_args,
        peft_config=peft_config,
    )

    trainer.model.print_trainable_parameters()
    trainer.train()

    save_path = f"{args.save_path}/{run_name}"
    trainer.save_model(save_path)
    logging.info("Best checkpoint saved to {}".format(save_path))

    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    args.model_name_or_path = save_path
    run_evaluation(args, trainer.model, tokenizer)


if __name__ == "__main__":
    main()
