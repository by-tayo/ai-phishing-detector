import joblib
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "../ml/phishing_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "../ml/vectorizer.pkl")

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

def predict_phishing(url: str):
    features = vectorizer.transform([url])
    prediction = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])
    return {
        "input": url,
        "is_phishing": bool(prediction),
        "confidence": round(float(confidence), 4),
    }