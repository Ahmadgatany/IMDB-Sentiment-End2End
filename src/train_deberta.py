import os
import yaml
import mlflow
import pandas as pd
from scipy.special import softmax
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)

# ========== Step 1: Load config.yaml ==========
with open("config/config.yaml", "r") as f:
    config = yaml.safe_load(f)

deberta_config = config["deberta"]
model_name = deberta_config["model"]["name"]
num_labels = deberta_config["model"]["num_labels"]
max_length = deberta_config["model"]["max_length"]
training_params = deberta_config["training"]

# ========== Step 2: Load raw dataset and preprocess ==========
raw_data_path = "Data/raw/IMDB Dataset.csv"
df = pd.read_csv(raw_data_path, on_bad_lines="skip", encoding="utf-8")


# Convert sentiment to label: pos → 1, neg → 0
df["label"] = df["sentiment"].map({"positive": 1, "negative": 0})
df = df[["review", "label"]].rename(columns={"review": "text"})

# Split into train / val / test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df["label"])
train_df, val_df = train_test_split(train_df, test_size=0.2, random_state=42, stratify=train_df["label"])

# ========== Step 3: Tokenization ==========
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

def tokenize_function(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=max_length)

train_dataset = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_dataset = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)

train_dataset = train_dataset.rename_column("label", "labels")
val_dataset = val_dataset.rename_column("label", "labels")
train_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])
val_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ========== Step 4: Model ==========
model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
model.to("cuda")

# ========== Step 5: TrainingArguments ==========
training_args = TrainingArguments(
    output_dir=training_params["output_dir"],
    evaluation_strategy=training_params.get("eval_strategy", "epoch"),
    save_strategy=training_params.get("save_strategy", "epoch"),
    learning_rate=float(training_params["learning_rate"]),
    per_device_train_batch_size=training_params["batch_size"],
    per_device_eval_batch_size=training_params["batch_size"],
    num_train_epochs=training_params["num_epochs"],
    weight_decay=training_params["weight_decay"],
    fp16=training_params.get("fp16", False),
    logging_dir=training_params["logging_dir"],
    metric_for_best_model=training_params.get("metric_for_best_model", "accuracy"),
    load_best_model_at_end=training_params.get("load_best_model_at_end", True),
    report_to=training_params.get("report_to", "none"),
    no_cuda=False
)

# ========== Step 6: Metrics ==========
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = logits.argmax(axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1": f1_score(labels, preds, average="weighted"),
    }

# ===================================================
import torch
print("🧠 Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

# ========== Step 7: MLflow logging and Training ==========
mlflow.set_experiment("DeBERTa Sentiment Classification")

with mlflow.start_run():
    mlflow.log_params({
        "model_name": model_name,
        "max_length": max_length,
        **training_params
    })

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics
    )

    trainer.train()

    # ========== Step 8: Save model ==========
    trainer.save_model(training_params["output_dir"])
    tokenizer.save_pretrained(training_params["output_dir"])

    print("✅ DeBERTa training complete. Model and tokenizer saved.")


# ========== Step 9: Prepare Test Set ==========
test_dataset = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)
test_dataset = test_dataset.rename_column("label", "labels")
test_dataset.set_format("torch", columns=["input_ids", "attention_mask", "labels"])

# ========== Step 10: Predictions ==========
predictions = trainer.predict(test_dataset)
logits = predictions.predictions
y_test_pred = logits.argmax(axis=-1)
y_test_probs = softmax(logits, axis=1)
y_test_true = predictions.label_ids

# Save results to CSV
results_df = pd.DataFrame({
    "review": test_df["text"],
    "true_label": y_test_true,
    "predicted_label": y_test_pred,
    "prob_negative": y_test_probs[:, 0],
    "prob_positive": y_test_probs[:, 1],
})

os.makedirs("Data/predictions", exist_ok=True)
results_df.to_csv("Data/predictions/deberta_preds.csv", index=False)

mlflow.log_artifact("Data/predictions/deberta_preds.csv")

print("📊 Test predictions saved for DeBERTa.")
