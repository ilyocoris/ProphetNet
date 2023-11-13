import tqdm
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

model = GPT2LMHeadModel.from_pretrained("recipes/gpt2/outputs/checkpoint-80000")
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

# load test src from data/eval/test.gpt2 and test tgt from data/eval/test.tgt
with open("data/eval/org_data/test.gpt2", "r") as f:
    test_src = f.readlines()
with open("data/eval/org_data/test.tgt", "r") as f:
    test_tgt = f.readlines()

# generate with max-length 64 and beam search 5 all the predictions from test_src
test_gen = []
for src in tqdm.tqdm(test_src):
    input_ids = tokenizer.encode(src, return_tensors="pt")
    outputs = model.generate(input_ids, max_length = 512, num_beams=5, num_return_sequences=1, no_repeat_ngram_size=2, early_stopping=True)
    for output in outputs:
        test_gen.append(tokenizer.decode(output, skip_special_tokens=True))

# save test_gen to data/eval/gpt2/gen_80000.txt
with open("data/eval/gpt2/gen_80000.txt", "w") as f:
    f.write("\n".join(test_gen))