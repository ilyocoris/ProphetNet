# CUDA_VISIBLE_DEVICES=2 python3 generate_recipes_gpt2.py
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    max_data = 1000
    n_return = 10
    dev_src_path = "data/raw/recipes/dev_ar.src"
    dev_tgt_path = "data/raw/recipes/dev_ar.tgt"
    run = "gpt2_lr5e-05"
    ##
    def load_from_lines(path):
        with open(path, 'r') as f:
            return [line.strip() for line in f]
    # load src
    dev_src = load_from_lines(dev_src_path)
    ##
    for checkpoint in ["10000", "20000", "30000", "40000", "50000", "60000", "70000", "80000", "90000", "100000", "110000", "120000", "130000", "140000", "150000", "160000", "170000", "180000", "190000", "200000", "210000", "220000", "230000", "240000", "250000", "260000", "270000", "280000", "290000", "300000", "310000", "320000", "330000", "340000", "350000", "360000", "370000"][::-1] :
        model_path = f"my_output/recipes/{run}/model/checkpoint-{checkpoint}"
        output_folder = f"my_output/recipes/{run}/gen/dev_ar_{checkpoint}"
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        model = AutoModelForCausalLM.from_pretrained(model_path)
        model.cuda()
        all_gens = []
        for sample in tqdm(dev_src[:max_data]):
            input_ids = tokenizer.encode(sample, return_tensors='pt')
            outputs = model.generate(
                input_ids.cuda(), 
                max_length=512+128,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                num_return_sequences=n_return,
                eos_token_id=tokenizer.eos_token_id
            )
            gens = []
            for i in range(n_return):     
                output = outputs[i, len(input_ids[0]):].tolist()
                gens.append(tokenizer.decode(output, skip_special_tokens=True))
        for i in range(n_return):
            with open(f"{output_folder}/{i}.gen", "w") as f:
                gens_i = [g[i] for g in all_gens]
                f.write("\n".join(gens_i))




if __name__ == '__main__':
    main()