import pandas as pd
from transformers import AutoTokenizer
from datasets import Dataset

def load_and_tokenize(filepath: str, model_name: str = "distilbert-base-uncased"):
    df = pd.read_csv(filepath)
    df.dropna(subset=["url", "label"], inplace=True)

    df["label"] = pd.to_numeric(df["label"])

    dataset = Dataset.from_pandas(df[["url", "label"]])
    dataset = dataset.train_test_split(test_size=0.2, seed=42)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize(batch):
        return tokenizer(
            batch["url"],
            padding="max_length",
            truncation=True,
            max_length=128
        )

    tokenized = dataset.map(tokenize, batched=True)
    return tokenized, tokenizer

if __name__ == "__main__":
    tokenized_data, tokenizer = load_and_tokenize("ml/data/final_dataset.csv")
    print(tokenized_data)