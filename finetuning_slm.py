import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, Trainer, TrainingArguments, RobertaTokenizerFast
import pandas as pd


# Define a custom dataset class
class HinglishDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(text, return_tensors='pt', truncation=True, padding='max_length', max_length=self.max_length)
        return {
            'input_ids': encoding['input_ids'].squeeze(),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': encoding['input_ids'].squeeze(),  # For language modeling
        }

# Load your tokenizer and model
model_save_path = 'tuned_hinglish_model'
tokenizer = RobertaTokenizerFast.from_pretrained(model_save_path)
model = AutoModelForCausalLM.from_pretrained(model_save_path)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load your text file
file_path = 'cleaned_hinglish_dataset.csv'
with open(file_path, 'r', encoding='utf-8') as file:
    hinglish_lines = file.readlines()
texts = hinglish_lines

# Create dataset
max_length = 128  # Adjust based on your use case
dataset = HinglishDataset(texts, tokenizer, max_length)

# Prepare DataLoader
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)  # Increased batch size
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Set checkpoint path in Google Drive
checkpoint_path = 'model_checkpoints'
os.makedirs(checkpoint_path, exist_ok=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir=checkpoint_path,                  # Directory to save model checkpoints
    run_name='hinglish_fine_tuning',             # Specify a unique run name
    overwrite_output_dir=True,                   # Overwrite output directory
    num_train_epochs=10,                         # Keep epochs as is
    per_device_train_batch_size=128,             # Increased batch size for training
    per_device_eval_batch_size=128,              # Increased batch size for evaluation
    gradient_accumulation_steps=2,               # Gradient accumulation (accumulating over 2 steps)
    eval_strategy="steps",                       # Keep eval strategy
    eval_steps=5000,                              # More frequent evaluation
    save_steps=5000,                              # Save checkpoints more frequently
    save_total_limit=3,                          # Limit total checkpoints
    logging_dir='./logs',                        # Directory for storing logs
    logging_steps=100,                           # Log every 100 steps
    learning_rate=3e-5,                          # Increased learning rate for larger batch size
    weight_decay=0.05,                           # Reduced weight decay
    lr_scheduler_type='cosine',                  # Cosine annealing learning rate schedule
    load_best_model_at_end=True,                 # Load best model at end of training
    metric_for_best_model="loss",                # Metric to determine best model
    report_to=['none'],
    fp16=True,                                   # Enable mixed precision training for faster processing
    dataloader_num_workers=4,                    # Increase the number of workers for data loading
)

# Create Trainer instance
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)

# Load from the last checkpoint if it exists
last_checkpoint = None
if os.path.exists(checkpoint_path) and os.listdir(checkpoint_path):
    # Check if there are any directories in checkpoint_path
    last_checkpoint_dirs = [os.path.join(checkpoint_path, d) for d in os.listdir(checkpoint_path) if os.path.isdir(os.path.join(checkpoint_path, d))]
    if last_checkpoint_dirs:
        last_checkpoint = sorted(last_checkpoint_dirs, key=os.path.getctime)[-1]

# Start fine-tuning
trainer.train(resume_from_checkpoint=last_checkpoint)

# Save final model and tokenizer
tuned_model_path = 'fine_tuned_hinglish_model'
model.save_pretrained(tuned_model_path)
tokenizer.save_pretrained(tuned_model_path)
print(f'Model and tokenizer saved to {tuned_model_path}')