from transformers import GPT2Tokenizer, GPT2LMHeadModel
from transformers import TextDataset, DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments

# Paths to your training and development data
train_file = "recipes/gpt2/data/train.txt"
dev_file = "recipes/gpt2/data/dev.txt"

# Load the tokenizer and model
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

# Create training and development datasets
train_dataset = TextDataset(tokenizer=tokenizer, file_path=train_file, block_size=128)
dev_dataset = TextDataset(tokenizer=tokenizer, file_path=dev_file, block_size=128)

# Define data collator for language modeling
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Define training arguments
training_args = TrainingArguments(
    output_dir="./recipes/gpt2/outputs",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    save_steps=20000,
    save_total_limit=6,
)

# Define Trainer and start training
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=dev_dataset,
)

trainer.train()

# Save model
trainer.save_model("./gpt2-finetuned")