import pandas as pd

def preprocess_phishtank(filepath: str, output_path: str):
    df = pd.read_csv(filepath, on_bad_lines="skip")
    df = df[df["verified"] == "yes"]
    df = df[["url", "target"]].copy()
    df["label"] = "phishing"
    df.dropna(subset=["url"], inplace=True)
    df.drop_duplicates(subset=["url"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")
    print(f"Total phishing samples: {len(df)}")
    return df

if __name__ == "__main__":
    preprocess_phishtank(
        "ml/data/phishing_information.csv",
        "ml/data/processed_phishing.csv"
    )
