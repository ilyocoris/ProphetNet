#!/bin/bash
# chmod +x generate.sh
# ./generate.sh

FILE_NAME="d1_uni"
STEP=20000
DATA_NAME="reverse"
DATA_PATH="data/raw/sequence"

CUDA_VISIBLE_DEVICES=6 torchrun --nproc_per_node=1 --nnodes=1 --master_port 29501 generate.py \
model.name='bert-base-uncased' batch_size=128 \
exp.name=$FILE_NAME load_step=$STEP \
data.name=$DATA_NAME data.path=$DATA_PATH tgt_len=32 max_pos_len=32 num_samples=50 \
intermediate_size=2048 num_attention_heads=8 dropout=0.2 \
in_channels=128 out_channels=128 time_channels=128 \
skip_sample=False gen_timesteps=2000 \
schedule_sampler='uniform' time_att=True att_strategy='txl' load_from_ema=False prediction=True \
fix_encoder=True model.custom_denoiser=True model.denoiser_layers=1 \
