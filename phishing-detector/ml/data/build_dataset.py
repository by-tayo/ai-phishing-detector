from pathlib import Path
from urllib.parse import urlsplit

import pandas as pd
from datasets import load_dataset

OUT = Path(__file__).resolve().parent / "final_dataset.csv"


def _split(url: str):
    try:
        return urlsplit(url if "://" in url else "http://" + url)
    except ValueError:  # malformed URLs (e.g. broken IPv6 brackets)
        return None


def host_of(url: str) -> str:
    parts = _split(url)
    return (parts.hostname or "").lower() if parts else ""


def has_path(url: str) -> bool:
    parts = _split(url)
    return bool(parts and (parts.path not in ("", "/") or parts.query))


def main() -> None:
    ds = load_dataset("pirocheto/phishing-url")
    df = pd.concat([ds[split].to_pandas()[["url", "status"]] for split in ds], ignore_index=True)

    df["url"] = df["url"].astype(str).str.strip()
    df = df[df["url"] != ""]
    df["label"] = (df["status"].str.lower() == "phishing").astype(int)
    df = df.drop_duplicates(subset="url").reset_index(drop=True)
    df["host"] = df["url"].map(host_of)
    df = df[df["host"] != ""][["url", "label", "host"]]
    df.to_csv(OUT, index=False)

    names = {0: "legitimate", 1: "phishing"}
    print(f"Saved {len(df):,} URLs to {OUT}\n")
    print(df["label"].map(names).value_counts().to_string())

    traits = pd.DataFrame({
        "has path": df["url"].map(has_path),
        "https": df["url"].str.lower().str.startswith("https://"),
        "www.": df["host"].str.startswith("www."),
    })
    summary = traits.groupby(df["label"].map(names)).mean().mul(100).round(1)
    print("\n% of URLs with each trait (the old dataset was 0% / 100% / 100% for legitimate):")
    print(summary.to_string())


if __name__ == "__main__":
    main()
