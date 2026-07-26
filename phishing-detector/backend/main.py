import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from schemas import PhishingRequest, PhishingResponse, EmailRequest, EmailResponse
from model import predict_phishing
from ml.phishtank_api import check_url_live
from ml.predict_transformer import predict_phishing as predict_phishing_transformer
from ml.predict_email import predict_email

app = FastAPI(
    title="AI-Powered Phishing Detector",
    description="Detects phishing URLs and emails using a trained ML model",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"status": "Phishing Detector API is running"}

@app.post("/predict", response_model=PhishingResponse)
def predict(request: PhishingRequest):
    result = predict_phishing(request.text)
    return PhishingResponse(
        input=result["input"],
        is_phishing=result["is_phishing"],
        confidence=result["confidence"],
    )

@app.post("/predict/transformer", response_model=PhishingResponse)
def predict_transformer(request: PhishingRequest):
    result = predict_phishing_transformer(request.text)
    return PhishingResponse(
        input=result["input"],
        is_phishing=result["is_phishing"],
        confidence=result["confidence"],
    )

@app.post("/check-live")
def check_live(request: PhishingRequest):
    result = check_url_live(request.text)
    return result

@app.post("/predict/email", response_model=EmailResponse)
def predict_email_route(request: EmailRequest):
    result = predict_email(request.raw_email)
    return EmailResponse(**result)