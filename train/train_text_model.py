from datasets import load_dataset, concatenate_datasets, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments

# Load main culture dataset
culture = load_dataset("deepkaria/indian-culture-dataset")["train"]

# Stream DRISHTIKON dataset
drishtikon_stream = load_dataset(
    "13ari/DRISHTIKON",
    split="train",
    streaming=True
)

# Take first 500 samples
drishtikon_samples = []
for i, sample in enumerate(drishtikon_stream):
    text = sample.get("text", "")
    if text:
        drishtikon_samples.append({"description": text})
    if i >= 500:
        break

drishtikon = Dataset.from_list(drishtikon_samples)

# Merge datasets
dataset = concatenate_datasets([culture, drishtikon])

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

def preprocess(example):
    text = example.get("description", "")
    return tokenizer(text, truncation=True, padding="max_length")

dataset = dataset.map(preprocess)

# Dummy labels
dataset = dataset.add_column("label", [0] * len(dataset))

model = AutoModelForSequenceClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2
)

training_args = TrainingArguments(
    output_dir="./culture_text_model",
    num_train_epochs=3,
    per_device_train_batch_size=8,
    save_strategy="no"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset
)

trainer.train()
trainer.save_model("./culture_text_model")
tokenizer.save_pretrained("./culture_text_model")
