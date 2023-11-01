### Download Model

`wget -O genie_ckpt "https://drive.google.com/u/0/uc?id=1-AZssEmgs0QdTp_w8-_4cPi0cV-Hot4N"`

### Generate
```
OUT_DIR = "data/outputs/"
MODEL_DIR = "genie_ckpt"
DATA_PATH = "data/exp1/"
DATA_NAME = "data"

python -u -m torch.distributed.launch --nproc_per_node=1 --master_port=9498 \
Genie_Generate.py \
--generate_path="data/outputs/" \
--eval_model_path="GENIE_ckpt-500w" \
--data_path="data/exp1/" \
--model_channels 128 --in_channel 128 --out_channel 128 --vocab_size 30522 \
--config_name="bert-base-uncased" --token_emb_type="random" \
--diffusion_steps 2000 --predict_xstart --noise_schedule="sqrt" \
--num_samples 5 --model_arch="s2s_CAT" --data_name="data" \
--training_mode="s2s" --tgt_max_len 64 --src_max_len 512 --batch_size=200 \
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