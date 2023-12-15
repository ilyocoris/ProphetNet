#!/bin/bash
# chmod +x train.sh
# ./train.sh


FILE_NAME="recipes_l6_te_c128"
DATA_PATH="data/raw"
DATA_NAME="recipes"
STEP=80000

CUDA_VISIBLE_DEVICES=3,4 torchrun --nproc_per_node=2 --nnodes=1 trainer_main.py \
model.name='bert-base-uncased' batch_size=64 grad_accum=3 \
total_steps=$STEP exp.name=$FILE_NAME \
data.name=$DATA_NAME data.path=$DATA_PATH tgt_len=128 max_pos_len=512 lr=8e-4 lr_step=40000 \
intermediate_size=2048 num_attention_heads=8 dropout=0.2 \
in_channels=128 out_channels=128 time_channels=128 \
eval_interval=3000 log_interval=1000 \
schedule_sampler='uniform' time_att=True att_strategy='txl' use_AMP=True \
fix_encoder=False model.custom_denoiser=False model.denoiser_layers=6
