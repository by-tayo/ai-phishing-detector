import os
import sys

ML_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "ml")
sys.path.insert(0, os.path.abspath(ML_DIR))

from preprocess import explain, load_url_model, phishing_probability  # noqa: E402

# One file holds the whole pipeline (URL cleanup, features and model),
# so there's no separate vectorizer to keep in sync.
model = load_url_model()


def predict_phishing(url: str):
    probability = phishing_probability(model, url)
    is_phishing = probability >= 0.5
    return {
        "input": url,
        "is_phishing": is_phishing,
        "confidence": round(probability if is_phishing else 1 - probability, 4),
        "reasons": explain(url),
    }
