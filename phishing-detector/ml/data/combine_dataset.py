import pandas as pd

# Load processed phishing URLs
phishing_df = pd.read_csv("ml/data/processed_phishing.csv")
phishing_df = phishing_df[["url"]].copy()
phishing_df["label"] = 1  # 1 = phishing

# Load Majestic Million legitimate URLs
legit_df = pd.read_csv("ml/data/majestic_million.csv")
legit_df["url"] = "https://www." + legit_df["Domain"]
legit_df = legit_df[["url"]].copy()
legit_df["label"] = 0  # 0 = legitimate

# Balance - same number of each
n = min(len(phishing_df), len(legit_df))
combined_df = pd.concat([
    phishing_df.sample(n, random_state=42),
    legit_df.sample(n, random_state=42)
], ignore_index=True)
combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
combined_df.to_csv("ml/data/final_dataset.csv", index=False)
print(f"Final dataset saved with {len(combined_df)} total samples")
print(combined_df["label"].value_counts())
