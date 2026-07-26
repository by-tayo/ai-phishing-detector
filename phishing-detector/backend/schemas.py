from pydantic import BaseModel

class PhishingRequest(BaseModel):
    text: str

class PhishingResponse(BaseModel):
    input: str
    is_phishing: bool
    confidence: float

class EmailRequest(BaseModel):
    raw_email: str

class LinkResult(BaseModel):
    url: str
    is_phishing: bool
    confidence: float

class EmailResponse(BaseModel):
    is_phishing: bool
    confidence: float
    text_model_confidence: float
    heuristic_score: float
    reasons: list[str]
    links: list[LinkResult]
    sender: str | None = None
    subject: str | None = None