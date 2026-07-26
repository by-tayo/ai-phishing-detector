import re
from email import message_from_string
from email.utils import parseaddr

URL_PATTERN = re.compile(r"https?://[^\s\"'<>\)]+", re.IGNORECASE)
ANCHOR_PATTERN = re.compile(r'<a\s[^>]*href=["\']([^"\']+)["\'][^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL)
IP_HOST_PATTERN = re.compile(r"^https?://\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")

URGENCY_PHRASES = [
    "verify your account",
    "confirm your password",
    "confirm your account",
    "account has been suspended",
    "account will be suspended",
    "account is suspended",
    "unusual activity",
    "unauthorized access",
    "click immediately",
    "act now",
    "urgent action required",
    "immediate action",
    "within 24 hours",
    "your account will be closed",
    "update your billing",
    "security alert",
    "suspicious login",
    "limited time",
    "avoid suspension",
    "verify your identity",
]

GENERIC_GREETINGS = [
    "dear customer",
    "dear user",
    "dear valued customer",
    "dear account holder",
    "dear member",
]

# A brand mentioned in the sender's display name should only appear alongside
# one of its real domains — otherwise it's a classic impersonation pattern.
BRAND_DOMAINS = {
    "paypal": ["paypal.com"],
    "amazon": ["amazon.com"],
    "apple": ["apple.com", "icloud.com"],
    "microsoft": ["microsoft.com", "outlook.com", "live.com"],
    "google": ["google.com", "gmail.com"],
    "netflix": ["netflix.com"],
    "bank of america": ["bankofamerica.com"],
    "wells fargo": ["wellsfargo.com"],
    "chase": ["chase.com"],
    "irs": ["irs.gov"],
    "docusign": ["docusign.com", "docusign.net"],
    "linkedin": ["linkedin.com"],
    "facebook": ["facebook.com", "fb.com"],
}


def extract_urls(text: str) -> list:
    return list(dict.fromkeys(URL_PATTERN.findall(text or "")))


def _strip_html_tags(text: str) -> str:
    return re.sub(r"<[^>]+>", " ", text or "")


def _extract_body(msg) -> str:
    if msg.is_multipart():
        parts = []
        for part in msg.walk():
            content_type = part.get_content_type()
            if content_type in ("text/plain", "text/html"):
                try:
                    payload = part.get_payload(decode=True)
                    charset = part.get_content_charset() or "utf-8"
                    parts.append(payload.decode(charset, errors="ignore"))
                except Exception:
                    parts.append(str(part.get_payload()))
        return "\n".join(parts)

    payload = msg.get_payload(decode=True)
    if payload is not None:
        try:
            charset = msg.get_content_charset() or "utf-8"
            return payload.decode(charset, errors="ignore")
        except Exception:
            pass
    return msg.get_payload() or ""


def _domain_of(address: str) -> str:
    return address.split("@")[-1].lower() if "@" in address else ""


def find_mismatched_links(html: str) -> list:
    mismatches = []
    for href, visible_text in ANCHOR_PATTERN.findall(html or ""):
        visible = _strip_html_tags(visible_text).strip()
        visible_urls = URL_PATTERN.findall(visible)
        if not visible_urls:
            continue
        visible_domain = _domain_of(visible_urls[0].split("/")[2]) if "://" in visible_urls[0] else ""
        try:
            href_domain = href.split("/")[2].lower() if "://" in href else href.lower()
        except IndexError:
            href_domain = href.lower()
        if visible_domain and href_domain and visible_domain not in href_domain:
            mismatches.append({"visible_text": visible, "actual_url": href})
    return mismatches


def analyze_email(raw_email: str) -> dict:
    raw_email = raw_email or ""
    msg = message_from_string(raw_email)

    from_header = msg.get("From", "") or ""
    reply_to_header = msg.get("Reply-To", "") or ""
    subject = msg.get("Subject", "") or ""

    from_name, from_addr = parseaddr(from_header)
    _, reply_addr = parseaddr(reply_to_header)

    from_domain = _domain_of(from_addr)
    reply_domain = _domain_of(reply_addr)

    body = _extract_body(msg).strip()
    has_headers = bool(from_header or reply_to_header or subject or body)
    if not body:
        # No parseable MIME structure (or no body part) — treat the whole
        # input as the body so plain pasted text still gets analyzed.
        body = raw_email

    body_lower = body.lower()
    reasons = []
    score = 0.0

    if reply_domain and from_domain and reply_domain != from_domain:
        reasons.append(
            f"Reply-To domain ({reply_domain}) differs from From domain ({from_domain})"
        )
        score += 0.25

    from_name_lower = from_name.lower()
    for brand, official_domains in BRAND_DOMAINS.items():
        if brand in from_name_lower and from_domain:
            if not any(from_domain == d or from_domain.endswith("." + d) for d in official_domains):
                reasons.append(
                    f"Sender name mentions '{brand.title()}' but the address domain is "
                    f"'{from_domain}', not an official {brand.title()} domain"
                )
                score += 0.3
                break

    urgency_hits = [phrase for phrase in URGENCY_PHRASES if phrase in body_lower]
    if urgency_hits:
        reasons.append(
            "Urgent/pressure language detected: " + ", ".join(urgency_hits[:5])
        )
        score += min(0.05 * len(urgency_hits), 0.25)

    if any(greeting in body_lower for greeting in GENERIC_GREETINGS):
        reasons.append("Generic greeting used instead of a personalized name")
        score += 0.1

    urls = extract_urls(body)

    mismatched_links = find_mismatched_links(body)
    if mismatched_links:
        reasons.append(
            f"{len(mismatched_links)} link(s) where the displayed text doesn't match the actual destination"
        )
        score += 0.2

    ip_links = [u for u in urls if IP_HOST_PATTERN.match(u)]
    if ip_links:
        reasons.append("Link points directly to an IP address instead of a domain")
        score += 0.2

    score = min(round(score, 4), 1.0)

    return {
        "score": score,
        "reasons": reasons,
        "from_address": from_addr or None,
        "reply_to": reply_addr or None,
        "subject": subject or None,
        "body": _strip_html_tags(body).strip(),
        "urls": urls,
        "has_headers": has_headers,
    }


if __name__ == "__main__":
    sample = (
        "From: PayPal Support <security@paypa1-verify.com>\n"
        "Reply-To: help@another-domain.net\n"
        "Subject: Urgent: Verify your account\n\n"
        "Dear Customer, your account has been suspended due to unusual activity. "
        "Please verify your account immediately: http://192.168.1.1/login"
    )
    print(analyze_email(sample))
