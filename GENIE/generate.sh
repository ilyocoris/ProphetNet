#!/bin/bash
# chmod +x generate.sh
# ./generate.sh

OUT_DIR="algorithmic/data/experiments/full/+/eval"
MODEL_DIR="algorithmic/data/experiments/full/+/model_checkpoint-80000"
DATA_PATH="algorithmic/data/raw/"
DATA_NAME="+"

CUDA_VISIBLE_DEVICES=3 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=9498 \
Genie_Generate.py \
--generate_path=$OUT_DIR \
--eval_model_path=$MODEL_DIR \
--data_path=$DATA_PATH \
--model_channels 128 --in_channel 128 --out_channel 128 --vocab_size 30522 \
--config_name="bert-base-uncased" --token_emb_type="random" \
--diffusion_steps 2000 --predict_xstart --noise_schedule="sqrt" \
--num_samples 1 --model_arch="s2s_CAT" --data_name=$DATA_NAME \
--training_mode="s2s" --tgt_max_len 16 --src_max_len 16 --batch_size=200 \
--interval_step 1 --seed 2023


# CUDA_VISIBLE_DEVICES=3 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=9498 \
# Genie_Generate.py \
# --generate_path="algorithmic/data/experiments/dev/+/eval" \
# --eval_model_path="algorithmic/data/experiments/dev/+/model_checkpoint-120000" \
# --data_path="algorithmic/data/raw/" \
# --model_channels 128 --in_channel 128 --out_channel 128 --vocab_size 126 \
# --config_name="bert-base-uncased" --token_emb_type="random" \
# --diffusion_steps 2000 --predict_xstart --noise_schedule="sqrt" \
# --num_samples 5 --model_arch="tiny_CAT" --data_name="+" \
# --training_mode="s2s" --tgt_max_len 16 --src_max_len 16 --batch_size=200 \
# --interval_step 1 --seed 2023