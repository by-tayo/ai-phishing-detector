import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

MODEL_DIR = "ml/models/phishing-bert"

model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
model.eval()

def evaluate(texts, labels):
    inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
    predictions = torch.argmax(outputs.logits, dim=1).numpy()

    print(classification_report(labels, predictions,
        target_names=["Legitimate", "Phishing"]))

    cm = confusion_matrix(labels, predictions)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=["Legitimate", "Phishing"],
        yticklabels=["Legitimate", "Phishing"])
    plt.title("Confusion Matrix - Phishing BERT")
    plt.tight_layout()
    plt.savefig("confusion_matrix_bert.png")
    print("Confusion matrix saved to confusion_matrix_bert.png")

if __name__ == "__main__":
    df = pd.read_csv("ml/data/final_dataset.csv").sample(500, random_state=42)
    texts = df["url"].tolist()
    labels = df["label"].tolist()
    evaluate(texts, labels)