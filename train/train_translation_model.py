# train_translation_local.py

import os
import zipfile
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, Trainer, TrainingArguments

# -----------------------------
# 1️⃣ Set paths
# -----------------------------
zip_path = "small_samanantar.zip"  # Your ZIP file in the project folder
extract_path = "small_samanantar"  # Folder to extract the ZIP

if not os.path.exists(extract_path):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)
    print(f"Extracted ZIP to {extract_path}")
else:
    print(f"Dataset folder {extract_path} already exists")

# -----------------------------
# 2️⃣ Languages
# -----------------------------
languages = ['as', 'bn', 'gu', 'hi', 'kn', 'ml', 'mr', 'or', 'pa', 'ta', 'te']

# -----------------------------
# 3️⃣ Load tokenizer & model
# -----------------------------
tokenizer = AutoTokenizer.from_pretrained("t5-small")
model = AutoModelForSeq2SeqLM.from_pretrained("t5-small")

# Freeze encoder for faster CPU training
for param in model.encoder.parameters():
    param.requires_grad = False
print("Encoder layers frozen. Only decoder will train.")

# -----------------------------
# 4️⃣ Function to load dataset per language pair
# -----------------------------
def load_pair_dataset(lang, max_samples=2000):
    pair_folder = os.path.join(extract_path, f"en-{lang}")
    src_file = os.path.join(pair_folder, f"train.{lang}")
    tgt_file = os.path.join(pair_folder, "train.en")

    with open(src_file, "r", encoding="utf-8") as f:
        src_lines = f.read().splitlines()
    with open(tgt_file, "r", encoding="utf-8") as f:
        tgt_lines = f.read().splitlines()

    n = min(len(src_lines), len(tgt_lines), max_samples)
    src_lines, tgt_lines = src_lines[:n], tgt_lines[:n]

    return Dataset.from_dict({
        "translation": [{"en": t, lang: s} for s, t in zip(src_lines, tgt_lines)]
    })

# -----------------------------
# 5️⃣ Preprocessing function
# -----------------------------
def preprocess(example):
    input_lang = [k for k in example["translation"].keys() if k != "en"][0]
    input_text = f"translate {input_lang} to en: {example['translation'][input_lang]}"
    target_text = example["translation"]["en"]

    model_inputs = tokenizer(
        input_text,
        truncation=True,
        padding="max_length",
        max_length=64
    )

    labels = tokenizer(
        target_text,
        truncation=True,
        padding="max_length",
        max_length=64
    )
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

# -----------------------------
# 6️⃣ Training loop per language pair
# -----------------------------
for lang in languages:
    print(f"\n=================== Training for en-{lang} ===================")

    # Load dataset
    dataset = load_pair_dataset(lang)

    # Preprocess
    dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

    # Training arguments
    output_dir = f"./translation_model_en-{lang}"
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=3,
        per_device_train_batch_size=8,
        save_strategy="no",
        logging_steps=50,
        fp16=False,  # CPU
        report_to=[]  # Disable WandB/TensorBoard if not needed
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset
    )

    # Train
    trainer.train()

    # Save model & tokenizer
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"✅ Model for en-{lang} saved at {output_dir}")
