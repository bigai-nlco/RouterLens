#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0

SEED=826
DATASET=nq-swap
MAX_SEQ_LENGTH=300
MODEL=allenai/OLMoE-1B-7B-0924-Instruct
TRAINING_METHOD=rt
EPOCH=3
LR=5e-4
MAX_NEW_TOKENS=32

python train.py \
    --seed ${SEED} \
    --dataset ${DATASET} \
    --max_seq_length ${MAX_SEQ_LENGTH} \
    --model_name_or_path ${MODEL} \
    --training_method ${TRAINING_METHOD} \
    --epoch ${EPOCH} \
    --learning_rate ${LR} \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 1 \
    --test_batch_size 8 \
    --max_new_tokens ${MAX_NEW_TOKENS} \
    --save_path saves/${DATASET} 
