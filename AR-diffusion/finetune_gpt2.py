# command line call:
# CUDA_AVALILABLE_DEVICES=1 python3 finetune_gpt2.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments
from datasets import load_dataset

def main():
    # Set the device to GPU if available
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 1
    learning_rate = 5e-5
    model_name = f"gpt2_lr{learning_rate}"
    # Load the pre-trained GPT-2 XL model and tokenizer
    model_id = "gpt2"
    tokenizer = GPT2Tokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(
        model_id,
        # device_map="auto"
    )
    # model.to(device)

    # Load your custom dataset from the datasets library
    dataset = load_dataset("corbt/all-recipes", split="train")  # Adjust as needed
    # dataset = dataset.select(range(1000)) 

    # Tokenize the dataset
    def tokenize_function(examples):
        return tokenizer(examples["input"])

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    # Create a data collator for language modeling
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Define output directories
    output_dir = f"./my_output/recipes/{model_name}/model"
    tensorboard_dir = f"./my_output/recipes/{model_name}/board"

    # Create output directories if they don't exist
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    # Define training arguments with additional configurations
    training_args = TrainingArguments(
        # torch_compile=True,
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=3,
        per_device_train_batch_size=8, # 200/16
        # gradient_accumulation_steps=4,
        save_steps=10000,
        save_total_limit=-1,
        logging_dir=tensorboard_dir,
        logging_steps=500,
        logging_first_step=True,
        learning_rate=learning_rate,
        save_strategy="steps",
        # save_freq=10000,
    )
    print("Training....")
    # Create Trainer and fine-tune the model
    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=tokenized_datasets,
    )

    trainer.train()

    # save model with name "last"
    trainer.save_model(output_dir + "/last")

if __name__ == "__main__":
    main()