#!/bin/bash
# chmod +x generate.sh
# ./generate.sh

OUT_DIR="data/mask_human/gen_20000"
DATA_PATH="recipes/"
DATA_NAME="mask_human"
CKPT_PATH="recipes/mask_human/outputs/model_checkpoint-20000"
 python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=9422 \
Genie_Generate.py \
--generate_path=$OUT_DIR \
--eval_model_path=$CKPT_PATH \
--data_path=$DATA_PATH \
--model_channels 128 --in_channel 128 --out_channel 128 --vocab_size 30522 \
--config_name="bert-base-uncased" --token_emb_type="random" \
--diffusion_steps 2000 --predict_xstart --noise_schedule="sqrt" \
--num_samples 5 --model_arch="s2s_CAT" --data_name=$DATA_NAME \
--training_mode="s2s" --tgt_max_len 64 --src_max_len 512 --batch_size=32 \
--interval_step 1 --seed 2023