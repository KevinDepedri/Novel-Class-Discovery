#!/usr/bin/env bash

python unbalanced_auto_novel.py \
        --warmup_model_dir $1 \
        --model_name $2 \
        --lr 0.1 \
        --gamma 0.1 \
        --weight_decay 1e-4 \
        --step_size 170 \
        --batch_size 128 \
        --epochs 200 \
        --rampup_length 50 \
        --rampup_coefficient 5.0 \
        --dataset_name cifar10 \
        --increment_coefficient 0.05 \
        --seed 0 \
        --mode train \
        --topk 5