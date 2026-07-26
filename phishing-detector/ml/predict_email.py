import os
import joblib

try:
    from ml.email_heuristics import analyze_email
except ImportError:
    from email_heuristics import analyze_email

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EMAIL_MODEL_PATH = os.path.join(BASE_DIR, "email_phishing_model.pkl")
EMAIL_VECTORIZER_PATH = os.path.join(BASE_DIR, "email_vectorizer.pkl")
URL_MODEL_PATH = os.path.join(BASE_DIR, "phishing_model.pkl")
URL_VECTORIZER_PATH = os.path.join(BASE_DIR, "vectorizer.pkl")

TEXT_WEIGHT = 0.5
LINK_WEIGHT = 0.3
HEURISTIC_WEIGHT = 0.2

_email_model = joblib.load(EMAIL_MODEL_PATH)
_email_vectorizer = joblib.load(EMAIL_VECTORIZER_PATH)
_url_model = joblib.load(URL_MODEL_PATH)
_url_vectorizer = joblib.load(URL_VECTORIZER_PATH)


def _phishing_probability(model, vectorizer, text: str) -> float:
    features = vectorizer.transform([text])
    proba = model.predict_proba(features)[0]
    phishing_index = list(model.classes_).index(1)
    return float(proba[phishing_index])


def predict_email(raw_email: str) -> dict:
    analysis = analyze_email(raw_email)

    text_score = _phishing_probability(_email_model, _email_vectorizer, analysis["body"] or raw_email)

    links = []
    link_score = 0.0
    for url in analysis["urls"]:
        confidence = _phishing_probability(_url_model, _url_vectorizer, url)
        links.append({
            "url": url,
            "is_phishing": confidence >= 0.5,
            "confidence": round(confidence, 4),
        })
        link_score = max(link_score, confidence)

    heuristic_score = analysis["score"]

    overall_score = (
        TEXT_WEIGHT * text_score
        + LINK_WEIGHT * link_score
        + HEURISTIC_WEIGHT * heuristic_score
    )
    overall_score = round(min(max(overall_score, 0.0), 1.0), 4)

    return {
        "is_phishing": overall_score >= 0.5,
        "confidence": overall_score,
        "text_model_confidence": round(text_score, 4),
        "heuristic_score": heuristic_score,
        "reasons": analysis["reasons"],
        "links": links,
        "sender": analysis["from_address"],
        "subject": analysis["subject"],
    }


if __name__ == "__main__":
    sample = (
        "From: PayPal Support <security@paypa1-verify.com>\n"
        "Reply-To: help@another-domain.net\n"
        "Subject: Urgent: Verify your account\n\n"
        "Dear Customer, your account has been suspended due to unusual activity. "
        "Please verify your account immediately: http://192.168.1.1/login"
    )
    print(predict_email(sample))
