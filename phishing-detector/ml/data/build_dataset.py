import pandas as pd
import urllib.request

# Load PhishTank data (all phishing)
phish_df = pd.read_csv("ml/data/phishing_information.csv")
phish_df = phish_df[["url"]].dropna()
phish_df["label"] = "phishing"

# Download Tranco top legitimate domains
print("Downloading Tranco legitimate URLS")
legit_url = "https://raw.githubusercontent.com/faizann24/Using-machine-learning-to-detect-malicious-URLs/master/data/good_urls.txt"
urllib.request.urlretrieve(legit_url, "ml/data/tranco.csv")

# Load and format Tranco data
tranco_df = pd.read_csv("ml/data/tranco.csv", header=None, names=["rank", "domain"])
tranco_df["url"] = "https://" + tranco_df["domain"]
tranco_df = tranco_df[["url"]]
tranco_df["label"] = "legitimate"

# Balance and combine
n = min(len(phish_df), len(tranco_df))
combined = pd.concat([phish_df.sample(n, random_state=42), tranco_df.sample(n, random_state=42)])
combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

# Save
combined.to_csv("ml/data/combined_dataset.csv", index=False)
print(f"Done! Saved {len(combined)} rows to ml/data/combined_dataset.csv")
print(combined["label"].value_counts())
