import torch
import os
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, Trainer, TrainingArguments, PreTrainedTokenizerFast, GPT2Config
from datasets import Dataset as HF_Dataset

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
        encoding = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding='max_length', 
            max_length=self.max_length
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),  # Remove batch dimension
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': encoding['input_ids'].squeeze(0),  # For causal language modeling
        }

# Load or create the tokenizer
tokenizer_path = "custom_tokenizer_files"
tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)

# Set the padding token
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

# Define the model configuration
config = GPT2Config(
    vocab_size=tokenizer.vocab_size,
    n_positions=1024,
    n_ctx=1024,
    n_embd=256,
    n_layer=10,
    n_head=8,
    activation_function="gelu_new",
    bos_token_id=50256,
    eos_token_id=50256,
    pad_token_id=50257,
    gradient_checkpointing=False,
    use_cache=True,
    max_position_embeddings=1024,
)

# Create the model from configuration
model = GPT2LMHeadModel(config)

# Update the model's embeddings to reflect the new vocabulary size
model.resize_token_embeddings(len(tokenizer))

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load raw dataset
file_path = 'Hinglish_Cleaned_Complete_Raw.txt'
with open(file_path, 'r', encoding='utf-8') as file:
    hinglish_lines = file.readlines()

texts = [line.strip() for line in hinglish_lines]

# Create dataset
max_length = 128  # Adjust max_length as per your requirements
dataset = HinglishDataset(texts, tokenizer, max_length)

# Split into training and validation datasets
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

# Prepare DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Set checkpoint path
checkpoint_path = 'model_checkpoints'
os.makedirs(checkpoint_path, exist_ok=True)

# Set training arguments
training_args = TrainingArguments(
    output_dir=checkpoint_path,
    run_name='hinglish_finetuning',
    overwrite_output_dir=True,
    num_train_epochs=15,               # Increased epochs for more training
    per_device_train_batch_size=128,   # Adjust batch size for better memory usage
    per_device_eval_batch_size=128,    # Adjust evaluation batch size
    gradient_accumulation_steps=2,     # Gradient accumulation to simulate a larger batch size
    eval_strategy="steps",             # Evaluation every few steps
    eval_steps=5000,                   # Evaluate every 5000 steps
    save_steps=5000,                   # Save the model every 5000 steps
    save_total_limit=3,                # Limit total checkpoints to 3
    logging_dir='./logs',              # Log directory
    logging_steps=100,                 # Log every 100 steps
    learning_rate=1e-5,                # Lower learning rate for finer tuning
    weight_decay=0.01,                 # Reduced weight decay for better convergence
    lr_scheduler_type='cosine',        # Continue using cosine annealing scheduler
    load_best_model_at_end=True,       # Load best model based on evaluation loss
    metric_for_best_model="loss",      # Use loss for determining best model
    report_to=['none'],
    fp16=True,                         # Mixed precision for faster training
    dataloader_num_workers=8,          # Increased number of workers for faster data loading
    warmup_steps=500,                  # Added warmup steps to prevent instability early on
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
    last_checkpoint_dirs = [os.path.join(checkpoint_path, d) for d in os.listdir(checkpoint_path) if os.path.isdir(os.path.join(checkpoint_path, d))]
    if last_checkpoint_dirs:
        last_checkpoint = sorted(last_checkpoint_dirs, key=os.path.getctime)[-1]
        print(f"Resuming from checkpoint: {last_checkpoint}")

# Start training
trainer.train(resume_from_checkpoint=last_checkpoint)


# Save final model and tokenizer
trained_model_path = 'pretrained_hinglish_model'
model.save_pretrained(trained_model_path)
tokenizer.save_pretrained(trained_model_path)
print(f'Model and tokenizer saved to {trained_model_path}')
