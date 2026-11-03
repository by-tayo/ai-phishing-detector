from pydantic import BaseModel, Field


class PhishingRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2048)


class PhishingResponse(BaseModel):
    input: str
    is_phishing: bool
    confidence: float
    reasons: list[str] = Field(default_factory=list)


class EmailRequest(BaseModel):
    raw_email: str = Field(..., min_length=1, max_length=50000)


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
