import pandas as pd
import re

def csv_cleaning(csv_file_path):
    df = pd.read_csv(csv_file_path)

    df.drop_duplicates(inplace=True)

    df['input'] = df['input'].str.lower()
    df['output'] = df['output'].str.lower()

    def clean_text(text):
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['input'] = df['input'].apply(clean_text)
    df['output'] = df['output'].apply(clean_text)

    df.dropna(inplace=True)

    df = df.drop_duplicates(subset=['input', 'output'])

    df.to_csv('cleaned_hinglish_dataset.csv', index=False)
    print("Data cleaning complete. Cleaned dataset saved as 'cleaned_hinglish_dataset.csv'.")

def txt_cleaning(txt_file_path):
    with open(txt_file_path, 'r') as file:
        data = file.readlines()

    cleaned_data = []
    for line in data:
        line = line.lower()
        line = re.sub(r'[^a-zA-Z0-9\s]', '', line)
        line = re.sub(r'\s+', ' ', line).strip()
        cleaned_data.append(line)

    with open('cleaned_hinglish_conversations.txt', 'w') as file:
        file.write('\n'.join(cleaned_data))

    print("Data cleaning complete. Cleaned dataset saved as 'cleaned_hinglish_conversations.txt'.")

csv_file_path = 'hinglish_conversations.csv'

txt_file_path = 'hinglish_conversations.txt'

# csv_cleaning(csv_file_path)

# txt_cleaning(txt_file_path)