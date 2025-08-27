#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

DATASET=nq-swap
MODEL=saves/nq-swap/OLMoE-1B-7B-0924-Instruct_router_e3_lr0.0005_sd826

python evaluation.py \
    --model_name_or_path ${MODEL} \
    --dataset ${DATASET} \
    --max_new_tokens 32 \
    --test_batch_size 8 
