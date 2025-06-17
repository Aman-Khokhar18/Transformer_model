

# 🌍 Transformer from Scratch in PyTorch

This repository implements the original **Transformer architecture** from the *"Attention is All You Need"* paper using PyTorch. It supports training a bilingual translation model from scratch, using HuggingFace datasets and custom tokenizers. 

Currently, it defaults to English-to-German translation using the `opus_books` dataset, but you can easily switch languages via the config file.

## 📚 Paper
> Vaswani et al., "Attention Is All You Need" (2017)  
> [Link to Paper](https://arxiv.org/abs/1706.03762)

---

## 🚀 Features

- Transformer encoder-decoder built from scratch
- Positional encoding, multi-head attention, and masking
- Greedy decoding for inference
- Trains on HuggingFace's multilingual datasets (`opus_books`)
- Custom `BilingualDataset` class with tokenizers
- Configurable language pairs, model size, and training hyperparameters
- Checkpoint saving and loading
- Easily extendable

---

## 🧾 Requirements

```bash
pip install torch torchvision torchaudio
pip install datasets tokenizers tqdm
````

---

## 🛠 Directory Structure

```
.
├── model.py                # Transformer model implementation
├── train.py                # Training script
├── dataset.py              # Dataset and masking logic
├── config.py               # Config file for hyperparameters
├── weights/                # Saved model weights
├── tokenizer_en.json       # Tokenizer for source language (auto-generated)
├── tokenizer_de.json       # Tokenizer for target language (auto-generated)
└── run/                    # Experiment directory
```

---

## ⚙️ Configuration

The `config.py` file contains all training and model settings. You can change the source and target languages like this:

```python
def get_config():
    return {
        ...
        "lang_src": "en",       # Source language
        "lang_tgt": "de",       # Target language
        "dataset_split": "de-en",  # HuggingFace dataset split
        ...
    }
```

### 🔄 Change Languages

This project uses HuggingFace's [`opus_books`](https://huggingface.co/datasets/opus_books) dataset.
To translate between other supported languages (e.g., French to English), modify:

```python
"lang_src": "fr",
"lang_tgt": "en",
"dataset_split": "en-fr",  # or "fr-en", depending on availability
```

---

## 🧪 Training the Model

Run the training script:

```bash
python train.py
```

This will:

* Download the dataset
* Train the Transformer model from scratch
* Save model weights in the `weights/` folder after each epoch
* Run validation after each epoch and print example translations

---

## 📤 Inference (Greedy Decoding)

The `run_validation()` function in `train.py` runs inference using greedy decoding and prints a few sample translations at the end of each epoch.

Example output:

```
------------ VALIDATION EXAMPLES ------------
SOURCE:     The story is about a boy who finds a dragon.
TARGET:     Die Geschichte handelt von einem Jungen, der einen Drachen findet.
PREDICTED:  Die Geschichte ist über einen Jungen, der einen Drachen findet.
--------------------------------------------
```

---

## 🧠 TODOs & Future Work

* Beam search decoding
* BLEU score evaluation
* Support for more datasets
* GUI interface or Gradio demo

---


## ✨ Acknowledgments

Inspired by the original [Attention is All You Need](https://arxiv.org/abs/1706.03762) paper.
Thanks to [HuggingFace](https://huggingface.co/) for datasets and tokenizer tools.
