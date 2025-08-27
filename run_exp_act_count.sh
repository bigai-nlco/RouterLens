#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

SEED=826
DATASET=nq-swap
MAX_LEN=300
MODEL=saves/nq-swap/OLMoE-1B-7B-0924-Instruct_router_e3_lr0.0005_sd826

python exp_act_count.py \
        --seed ${SEED} \
        --model_name_or_path ${MODEL} \
        --dataset ${DATASET} \
        --max_sample 2000 \
        --max_seq_length ${MAX_LEN}
