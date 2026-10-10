"""Score a URL with the trained model from the command line.

    python3 ml/predict.py "https://www.paypa1-secure-login.com"
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import explain, load_url_model, phishing_probability  # noqa: E402

model = load_url_model()


def predict_phishing(text: str) -> dict:
    probability = phishing_probability(model, text)
    is_phishing = probability >= 0.5
    return {
        "input": text,
        "is_phishing": is_phishing,
        "confidence": round(probability if is_phishing else 1 - probability, 4),
        "reasons": explain(text),
    }


if __name__ == "__main__":
    for url in sys.argv[1:] or ["http://suspicious-login.verify-account.com"]:
        print(predict_phishing(url))
