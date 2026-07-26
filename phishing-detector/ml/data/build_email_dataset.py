import pandas as pd
from datasets import load_dataset

LABEL_MAP = {
    "Phishing Email": 1,
    "Safe Email": 0,
}


def build_email_dataset(output_path: str = "ml/data/email_dataset.csv"):
    dataset = load_dataset("zefang-liu/phishing-email-dataset", split="train")
    df = dataset.to_pandas()

    df = df.rename(columns={"Email Text": "text", "Email Type": "label"})
    df = df[["text", "label"]]

    df.dropna(subset=["text", "label"], inplace=True)
    df["label"] = df["label"].map(LABEL_MAP)
    df.dropna(subset=["label"], inplace=True)
    df["label"] = df["label"].astype(int)

    df.drop_duplicates(subset=["text"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(output_path, index=False)
    print(f"Email dataset saved to {output_path}")
    print(f"Total samples: {len(df)}")
    print(df["label"].value_counts())
    return df


if __name__ == "__main__":
    build_email_dataset()
