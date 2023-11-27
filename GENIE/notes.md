### Environment for 3090 (hrist)

```pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118```


### Download Model

`wget -O genie_ckpt "https://drive.google.com/u/0/uc?id=1-AZssEmgs0QdTp_w8-_4cPi0cV-Hot4N"`

### Generate
```
OUT_DIR="data/mask_human/gen_80000_tests"
DATA_PATH="recipes/"
DATA_NAME="mask_human"
CKPT_PATH="recipes/mask_human/outputs/model_checkpoint-80000"

CUDA_VISIBLE_DEVICES=1 python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=9499 \
Genie_Generate.py \
--generate_path="data/mask_human/gen_800000_tests/variability" \
--eval_model_path="recipes/mask_30/outputs/model_checkpoint-80000" \
--data_path="data/" \
--model_channels 128 --in_channel 128 --out_channel 128 --vocab_size 30522 \
--config_name="bert-base-uncased" --token_emb_type="random" \
--diffusion_steps 2000 --predict_xstart --noise_schedule="sqrt" \
--num_samples 100 --model_arch="s2s_CAT" --data_name="eval" \
--training_mode="s2s" --tgt_max_len 64 --src_max_len 512 --batch_size=128 \
--interval_step 1 --seed 2023
```

### Finetune
```
OUT_DIR = "/Your/output/path"
DATA_PATH = "/Your/data/path"
DATA_NAME = "xsum_data"
PRETRAIN_CKPT_PATH = "/Your/pretrain_ckpt/path"


python -u -m torch.distributed.launch --nproc_per_node=4 --master_port=9421 \
./GENIE_main/Genie_Finetune.py \
--checkpoint_path=$OUT_DIR \
--model_channels 128 --in_channel 128 --out_channel 128 --vocab_size 30522 \
--config_name="bert-base-uncased" --token_emb_type="random" --model_arch="s2s_CAT" \
--diffusion_steps 2000 --predict_xstart --noise_schedule="sqrt" --training_mode="s2s" \
--schedule_sampler="loss-second-moment" --tgt_max_len 64 --src_max_len 512 --data_name=$DATA_NAME \
--data_path=$DATA_PATH \
--lr_anneal_steps 120000 --batch_size 64 --lr 5e-05 --warmup_steps 7200 --train_type="S2S_Diffusion" \
--eval_interval 200 --log_interval 200 --save_interva 20000 \
--pretrain_model_path=$PRETRAIN_CKPT_PATH
```