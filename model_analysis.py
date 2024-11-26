import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, PreTrainedTokenizerFast
from tqdm import tqdm  # For progress bar during evaluation

# Load the fine-tuned model and tokenizer
fine_tuned_model_path = 'fine_tuned_hinglish_model'
model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_path)
tokenizer = PreTrainedTokenizerFast.from_pretrained(fine_tuned_model_path)

# Move model to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# Load the entire dataset (same as before)
file_path = 'cleaned_hinglish_conversations.txt'  # Path to your dataset
with open(file_path, 'r', encoding='utf-8') as file:
    full_dataset_lines = file.readlines()

texts = [line.strip() for line in full_dataset_lines]

# Tokenize the entire dataset
max_length = 128  # Adjust max_length as needed
encodings = tokenizer(
    texts, 
    truncation=True, 
    padding='max_length', 
    max_length=max_length, 
    return_tensors='pt'
)

# Create a DataLoader for evaluation (no shuffle here)
eval_dataset = torch.utils.data.TensorDataset(
    encodings['input_ids'], 
    encodings['attention_mask'], 
    encodings['input_ids']  # Same as labels for language modeling
)

eval_loader = DataLoader(eval_dataset, batch_size=32)

# Set the model to evaluation mode
model.eval()

# Evaluate the model on the entire dataset
total_loss = 0
num_batches = 0

with torch.no_grad():  # Disable gradient computation during evaluation
    for batch in tqdm(eval_loader, desc="Evaluating"):
        input_ids = batch[0].to(device)  # Move input_ids to GPU
        attention_mask = batch[1].to(device)  # Move attention_mask to GPU
        labels = batch[2].to(device)  # Move labels to GPU

        # Forward pass to get the loss
        outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_loss += loss.item()
        num_batches += 1

# Compute the average loss
avg_loss = total_loss / num_batches
print(f'Average loss over the entire dataset: {avg_loss}')

# Compute Perplexity (optional, as an additional metric)
import math
perplexity = math.exp(avg_loss)
print(f'Perplexity: {perplexity}')
