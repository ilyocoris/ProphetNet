#!/bin/bash
# chmod +x algorithmic.sh
# ./algorithmic.sh

OUT_DIR="algorithmic/data/experiments/dev/+"
DATA_PATH="algorithmic/data/raw/"
DATA_NAME="+"
# PRETRAIN_CKPT_PATH="GENIE_ckpt-500w"

CUDA_VISIBLE_DEVICES=3,7 python -u -m torch.distributed.launch --nproc_per_node=2 --master_port=9421 \
Genie_Algorithmic.py \
--checkpoint_path=$OUT_DIR \
--model_channels 128 --in_channel 128 --out_channel 128 --vocab_size 126 \
--config_name="bert-base-uncased" --token_emb_type="random" --model_arch="tiny_CAT" \
--diffusion_steps 2000 --predict_xstart --noise_schedule="sqrt" --training_mode="s2s" \
--schedule_sampler="loss-second-moment" --tgt_max_len 16 --src_max_len 16 --data_name=$DATA_NAME \
--data_path=$DATA_PATH \
--lr_anneal_steps 120000 --batch_size 32 --lr 5e-05 --warmup_steps 7200 --train_type="S2S_Diffusion" \
--eval_interval 200 --log_interval 200 --save_interval 20000 
# --pretrain_model_path=$PRETRAIN_CKPT_PATH