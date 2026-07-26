import os
os.environ["CUDA_VISIBLE_DEVICES"] = ""

from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models", "phishing-bert")

_model = None
_tokenizer = None


def _load_model():
    global _model, _tokenizer
    if _model is None:
        # Loaded lazily so the rest of the API (URL/email endpoints) still
        # works even when the BERT weights aren't present locally (they're
        # gitignored — see README) and only this endpoint fails.
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR)
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        _model.eval()
    return _model, _tokenizer


def predict_phishing(text: str):
    model, tokenizer = _load_model()
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=128
    )
    with torch.no_grad():
        outputs = model(**inputs)
    probabilities = torch.softmax(outputs.logits, dim=1)
    prediction = torch.argmax(probabilities, dim=1).item()
    confidence = probabilities[0][prediction].item()
    return {
        "input": text,
        "is_phishing": bool(prediction),
        "confidence": round(confidence, 4)
    }

if __name__ == "__main__":
    test_url = "http://secure-login.verify-account.phishing.com"
    result = predict_phishing(test_url)
    print(result)