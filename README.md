# Tiny Hinglish Model

This repository contains the Python scripts used to create a custom, tiny Hinglish-speaking model based on the GPT-2 architecture. The model consists of 21 million parameters and is trained to generate responses in Hinglish for general everyday conversations. Below is the step-by-step process of how the model was developed.

---
**[Try it out now!](https://huggingface.co/spaces/Abhishekcr448/Hinglish-Chat-Prediction)**

**Access Model:** [Abhishekcr448/Tiny-Hinglish-Chat-21M](https://huggingface.co/Abhishekcr448/Tiny-Hinglish-Chat-21M)

---

### Project Overview

This project aimed to create a small text-generative model in Hinglish using the GPT-2 architecture. The model is trained to predict replies and responses in Hinglish on general everyday conversational topics. The project included creating a custom tokenizer, training the model from scratch, and fine-tuning it using a relevant conversational dataset.

---

### Table of Contents

1. [Tokenizer Creation](#tokenizer-creation)
2. [Pre-training the Model](#pre-training-the-model)
3. [Fine-tuning the Model](#fine-tuning-the-model)
4. [Model Evaluation and Results](#model-evaluation-and-results)
5. [Cost Analysis](#cost-analysis)
6. [Lessons Learned](#lessons-learned)
7. [Acknowledgements](#acknowledgements)
8. [Future Improvements](#future-improvements)

---

### Tokenizer Creation

1. **Datasets Used**:
   - A combined dataset of 1.7 million records:
     - 700k records from various Hinglish datasets extracted from HuggingFace.
     - 1 million records generated using GPT-4 API (batch processing method for everyday conversations). [Access the dataset](https://huggingface.co/spaces/Abhishekcr448/Hinglish-Chat-Prediction)
   
2. **Cleaning the Data**:
   - The datasets were cleaned by removing unnecessary characters and converting everything to lowercase using the cleaning script (`clean_data.py`).

3. **Tokenization**:
   - The data was tokenized using a custom BPE tokenizer created with the script `custom_tokenizer.py`.
   - The tokenizer outputted 3 major files: `tokenizer.json`, `merges.txt`, and `vocab.json`.

   **Note**: The tokenizer format might change during training, you can replace it with the original `tokenizer.json` created in Step 1.

---

### Pre-training the Model

1. **Model Training**:
   - The pre-training was conducted on a custom dataset using the script `pretraining.py` for 20 epochs.
   - Checkpoints were saved every 5000 steps to allow for training interruptions.

2. **Resources**:
   - The model was trained on Vast.ai using an RTX 4090 GPU with 24GB VRAM, at a rate of $0.2 per hour (4 hours).
   - Google Drive cloud was connected to Vast.ai to save checkpoints and the final model.

3. **Output Files**:
   - After pre-training, the following files were generated:
     - `config.json`
     - `generation_config.json`
     - `model.safetensors`
     - `special_tokens_map.json`
     - `tokenization_config.json`
     - `tokenizer.json`

---

### Fine-tuning the Model

1. **Fine-tuning Process**:
    - Fine-tuning was done using my everyday conversation dataset (`conversations_dataset.txt`).
    - The fine-tuning script `fine_tuning_slm.py` was used, and training was performed for up to 20 epochs.

2. **Model Output**:
   - The final fine-tuned model of size 80mb was uploaded to HuggingFace.

   **Note**: The tokenizer format might change during training, you can replace it with the original `tokenizer.json` created in Step 1.

---

### Model Evaluation and Results

**Training Metrics for Different Epochs**:
- **Model with 5 epochs**:
  - Average Loss: 0.6177
  - Perplexity: 1.8547
- **Model with 10 epochs**:
  - Average Loss: 0.6093
  - Perplexity: 1.8391
- **Model with 15 epochs**:
  - Average Loss: 0.6037
  - Perplexity: 1.8289
- **Model with 20 epochs**:
  - Average Loss: 0.5976
  - Perplexity: 1.8177
- **Model with 30 epochs** (not recommended):
  - Average Loss: 0.6447
  - Perplexity: 1.9053
- **Small model (5 epochs)** (6-8 layers, poor performance, not uploaded):
    - Average Loss: 0.7309
    - Perplexity: 2.0770

---

### Cost Analysis

- **Dataset Creation**: $15 (for generating and cleaning datasets).
- **GPU Usage**: $10 (for 4 hours of training on Vast.ai).
- **Total Estimated Cost**: $25.

   If I had avoided mistakes and unnecessary other model creation, the project could have been completed for $15–$20.

---

### Lessons Learned

1. **Data Quality & Size**:
   - High-quality data with relevant context is key. Even 5–10 epochs of training can yield good results with the right data.
   
2. **Model Configuration**:
   - Experimenting with smaller models led to poor performance (higher loss and perplexity). Sticking to the original model architecture is recommended.

---

### Acknowledgements

- The inspiration for this project came from the [Tiny Stories](https://huggingface.co/roneneldan/TinyStories-33M) model on HuggingFace.
- I used HuggingFace datasets and GPT-4 API to generate the everyday conversational dataset.

---

### Future Improvements

- The main goal was to integrate this model into a mobile app for real-time conversation response predictions. If anyone successfully does this, please let me know, and I’d love to appreciate your work!
- Improvements in fine-tuning and better model compression methods are always welcome.
- Contributions and suggestions to make the model more efficient or to improve training scripts are encouraged.

---

Feel free to explore the scripts, try them out, and contribute to the project. Happy coding!
