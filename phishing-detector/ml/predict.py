import joblib

model = joblib.load("phishing_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

def predict_phishing(text: str):
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    confidence = max(model.predict_proba(features)[0])
    
    return {
        "input": text,
        "is_phishing": bool(prediction),
        "confidence": round(float(confidence), 4),
    }

if __name__ == "__main__":
    test_url = "http://suspicious-login.verify-account.com"
    print(predict_phishing(test_url))