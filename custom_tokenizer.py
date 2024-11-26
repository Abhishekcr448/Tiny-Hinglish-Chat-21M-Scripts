from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
from tokenizers.normalizers import NFD
import os

# Step 1: Load your combined dataset
# The dataset should be a plain text file where each line is a sentence.
with open('Hinglish_Cleaned_Complete_Raw.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# Step 2: Initialize a BPE Tokenizer
tokenizer = Tokenizer(models.BPE())

# Step 3: Add normalizers to pre-process the text
tokenizer.normalizer = NFD()  # Normalizing accents
tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()  # Tokenize by whitespace

# Step 4: Create a Trainer
trainer = trainers.BpeTrainer(
    vocab_size=50257,  # Adjust based on your requirements
    min_frequency=2,   # Minimum frequency of token occurrence
    special_tokens=[
        "[PAD]", "[UNK]", "[BOS]", "[EOS]", "[MASK]", "[SEP]"
    ],
)

# Step 5: Train the Tokenizer
tokenizer.train_from_iterator(lines, trainer=trainer)

# Step 6: Save the tokenizer
save_folder = "custom_tokenizer_files"
os.makedirs(save_folder, exist_ok=True)

# Save the vocabulary and merges files in the new folder
tokenizer.save(save_folder+"/tokenizer.json")
tokenizer.model.save(save_folder)
print("Tokenizer saved successfully!")

# Load the custom tokenizer
tokenizer = Tokenizer.from_file("custom_tokenizer_files/tokenizer.json")

# Test the tokenizer
output = tokenizer.encode("aaj ka weather bada mazedaar hai na haa bas thoda garam hai")
print(output.tokens)  # This will show the tokens created by the tokenizer


