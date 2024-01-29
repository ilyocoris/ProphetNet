# CUDA_VISIBLE_DEVICES=5 python3 generate_recipes.py
import os
import torch
import hydra
import evaluate
from tqdm import tqdm
from functools import partial
# import matplotlib.pyplot as plt
import torch.distributed as dist
from transformers import AutoTokenizer, set_seed
from torch.utils.data import DataLoader



from utils import load_states_from_checkpoint
from data_utils.s2s_dataset import load_jsonl_data, S2S_dataset
from model_utils.create_model import create_model, create_gaussian_diffusion
from generate import denoised_fn_round

def initialize_distributed():
    if not dist.is_initialized():
        # Initialize the distributed environment
        dist.init_process_group(backend='gloo')  # 'gloo' is suitable for local development



def main():
    device = 0
    task = "recipes"
    run = "eb6ah8_d6_c128_lr2e-4_v2"
    config_file = "config.yaml"
    max_data = 1000
    test_tgt_path = "data/raw/recipes/dev.tgt"
    test_src_path = "data/raw/recipes/dev.src"
    batch_size = 128
    is_ema = True

    # Fake distributed
    # Call the initialization function
    os.environ['RANK'] = '0'
    os.environ['WORLD_SIZE'] = '1' 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12345'
    initialize_distributed()

    # Load config and tokenizer
    if hydra.core.global_hydra.GlobalHydra.instance() is not None:
        hydra.core.global_hydra.GlobalHydra.instance().clear()
    hydra.initialize(config_path=f"confs")
    config = hydra.compose(config_name=config_file)

    for checkpoint in ["120000", "130000", "140000", "150000", "160000", "170000", "180000", "190000", "200000", "10000", "20000", "30000", "40000", "50000", "60000", "70000", "80000", "90000", "100000", "110000"] :

        # Load Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config.tokenizer.name_or_path)
        vocab_size = tokenizer.vocab_size

        # Load Model
        if is_ema:
            eval_model_path = f"my_output/{task}/{run}/model/ema_0.9999_checkpoint-{checkpoint}"
        else:
            eval_model_path = f"my_output/{task}/{run}/model/model_checkpoint-{checkpoint}"
        diffusion = create_gaussian_diffusion(config)
        model = create_model(config, vocab_size)
        model_saved_state = load_states_from_checkpoint(eval_model_path, dist.get_rank())
        model.load_state_dict(model_saved_state.model_dict)
        if config.ddim_sample:
            sample_fn = (diffusion.ddim_sample_loop)
        else:
            sample_fn = (diffusion.p_sample_loop)
        emb_model = model.word_embedding
        model.to(device)

        print(f"Loaded model from {eval_model_path}")

        # Load Data
        test_data = []
        with open(test_src_path, "r") as f_src , open(test_tgt_path, "r") as f_tgt:
            for src, tgt in zip(f_src, f_tgt):
                test_data.append({"src":src.strip(), "tgt":tgt.strip()})
        if max_data:
            test_data = test_data[:max_data]

        dev_dataset = S2S_dataset(test_data, tokenizer, config)
        dev_dataloader = DataLoader(
            dev_dataset, batch_size=batch_size, 
            drop_last=False, pin_memory=True, num_workers=config.num_workers, 
            collate_fn=S2S_dataset.get_collate_fn(config),
            shuffle=False,
        )
        for seed in [92, 376, 101, 12, 55, 11111, 87, 1234, 4321, 1]:
            print(f"***** Generating for seed {seed}: *****")
            set_seed(seed)
            # Generate
            each_sample_list = []

            for _, batch in enumerate(tqdm(dev_dataloader)):
                with torch.no_grad():
                    encoder_hidden_states = model.encoder(
                        input_ids=batch['src_input_ids'].to(device), 
                        attention_mask=batch['src_attention_mask'].to(device),
                    ).last_hidden_state  # [bs, seq_len, hz]

                if config.pred_len:
                    with torch.no_grad():
                        length_out = model.get_pred_len(
                            encoder_hidden_states=encoder_hidden_states,
                            src_masks=batch['src_attention_mask'].to(device),
                            normalize=True,
                        )  # [bs, max_pos_len]
                        pred_lengs = length_out.max(-1)[1]  # [bs,], max return tuple(value, indices)

                    tgt_attention_mask = []
                    for len_item in pred_lengs:
                        tgt_attention_mask.append([1] * len_item + [0] * (max(pred_lengs) - len_item))
                    tgt_attention_mask = torch.tensor(tgt_attention_mask).long()
                    
                    input_shape = (
                        tgt_attention_mask.shape[0], tgt_attention_mask.shape[1], config.in_channels,
                    )
                else:
                    pred_lengs, tgt_attention_mask = None, None
                    input_shape = (
                        batch['src_input_ids'].shape[0], config.tgt_len, config.in_channels,
                    )

                model_kwargs = {'src_attention_mask': batch['src_attention_mask'].to(device),
                                'tgt_attention_mask': tgt_attention_mask,
                                'encoder_hidden_states': encoder_hidden_states,}
                sample = sample_fn(
                    model,
                    input_shape,
                    clip_denoised=config.clip_denoised,
                    # "Freeze" some parameters for easy recall.
                    denoised_fn=partial(denoised_fn_round,
                                        config, emb_model.to(device)),
                    progress=True,
                    model_kwargs=model_kwargs,
                    pred_lengs=pred_lengs,
                    top_p=-1.0,
                )


                logits = model.get_logits(sample)  # (bs, seq_len, vocab_size)
                sample_id_tensor = torch.argmax(logits, dim=-1)
                generations = tokenizer.batch_decode(sample_id_tensor, skip_special_tokens=True)
                each_sample_list.extend(generations)
                print(generations[:2], end="\n***")

            # Save Results
            # create gen folder if it does not exist
            gen_folder_path = f"my_output/{task}/{run}/gen/dev_{checkpoint}" if not is_ema else f"my_output/{task}/{run}/gen/dev_{checkpoint}_ema"
            if not os.path.exists(gen_folder_path):
                os.makedirs(gen_folder_path)
            # save each_sample_list to my_output/recipes/eb6_d6_c128_wd01/gen/dev.gen
            with open(f"{gen_folder_path}/{seed}.gen", "w") as f:
                for item in each_sample_list:
                    f.write(item+"\n")

            # Bonus Rouge Score
            with open(f"{gen_folder_path}/{seed}.gen", "r") as f:
                gen = f.readlines()
            golds = [t["tgt"] for t in test_data[:len(gen)]]
            rouge = evaluate.load('rouge')
            scores = rouge.compute(
                predictions=[g.strip() for g in gen],
                references=golds,
            )
            print(f"***** Results for seed {seed}: *****")
            print(scores)

if __name__ == "__main__":
    main()