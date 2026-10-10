"""URL preprocessing shared by training and the API.

train.py, predict.py, predict_email.py and the backend all use this module, so
a URL is turned into features exactly the same way at training time and at
prediction time. The saved model (phishing_model.pkl) refers to these
functions as `preprocess.<name>`, so anything that loads the model must put
the ml/ folder on sys.path first (see load_url_model below).
"""
from __future__ import annotations

import ipaddress
import math
import re
from collections import Counter
from pathlib import Path
from urllib.parse import urlsplit

import joblib
import numpy as np
import tldextract

# Offline: use the public-suffix snapshot bundled with tldextract, no downloads.
_EXTRACT = tldextract.TLDExtract(suffix_list_urls=())

BRANDS = [
    "paypal", "amazon", "apple", "icloud", "microsoft", "outlook", "office365",
    "google", "gmail", "netflix", "bankofamerica", "wellsfargo", "chase",
    "irs", "docusign", "linkedin", "facebook", "instagram", "whatsapp",
    "dropbox", "adobe", "yahoo", "coinbase", "binance", "usps", "fedex", "dhl",
]
# Common character swaps in lookalike domains (paypa1, g00gle, micr0soft).
_LOOKALIKE = str.maketrans({"0": "o", "1": "l", "3": "e", "4": "a", "5": "s", "7": "t", "@": "a"})
LURE_WORDS = [
    "login", "log-in", "signin", "sign-in", "verify", "verification", "secure",
    "account", "update", "confirm", "banking", "password", "wallet", "unlock",
    "suspend", "billing", "invoice", "support", "webscr", "auth",
]
RISKY_TLDS = {
    "xyz", "top", "club", "online", "site", "live", "icu", "buzz", "shop",
    "work", "click", "link", "rest", "cfd", "sbs", "monster", "tk", "ml", "ga", "cf", "gq",
}

# Domains where anyone can publish a page. A URL on these is never treated as
# "well-known", even though the domains themselves are popular.
SHARED_HOSTING = {
    "blogspot.com", "github.io", "githubusercontent.com", "web.app", "firebaseapp.com",
    "appspot.com", "amazonaws.com", "cloudfront.net", "googleapis.com", "azurewebsites.net",
    "windows.net", "herokuapp.com", "netlify.app", "vercel.app", "pages.dev", "workers.dev",
    "r2.dev", "glitch.me", "repl.co", "wixsite.com", "weebly.com", "wordpress.com",
    "squarespace.com", "godaddysites.com", "webflow.io", "000webhostapp.com", "sharepoint.com",
    "ngrok.io", "ngrok-free.app", "duckdns.org", "ddns.net", "no-ip.com", "ipfs.io", "dweb.link",
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "forms.gle", "is.gd", "cutt.ly", "rebrand.ly",
    "translate.goog", "dropbox.com", "box.com", "wetransfer.com", "1drv.ms", "notion.site",
    "framer.app", "carrd.co", "jimdosite.com", "myshopify.com", "linktr.ee", "typeform.com",
}
# Popular sites with sections where anyone can publish (forms, docs, sites).
SHARED_HOSTS = {
    "sites.google.com", "docs.google.com", "drive.google.com", "forms.office.com",
    "onedrive.live.com", "storage.cloud.google.com",
}

FEATURE_NAMES = [
    "host_length", "path_length", "query_params", "subdomain_count", "host_dots",
    "host_hyphens", "host_digits", "host_entropy", "is_ip", "has_at", "has_punycode",
    "risky_tld", "brand_not_registered_domain", "lookalike_brand", "lure_words",
    "has_port", "double_slash_in_path", "url_length",
]


def _parts(url: str):
    url = (url or "").strip()
    try:
        return urlsplit(url if "://" in url else "http://" + url)
    except ValueError:
        return urlsplit("http://invalid")


def normalize(url: str) -> str:
    """Lowercase and drop the scheme, a leading 'www.' and a trailing '/'.

    In the training data, https and 'www.' are much more common on safe URLs,
    so leaving them in would let the model treat 'https://www.' as a sign of
    safety. Phishing sites can add both for free.
    """
    url = (url or "").strip().lower()
    url = re.sub(r"^[a-z][a-z0-9+.-]*://", "", url)
    url = re.sub(r"^www\d?\.", "", url)
    return url.rstrip("/")


def normalize_many(urls) -> list[str]:
    return [normalize(u) for u in urls]


def registered_domain(url: str) -> str:
    """'login.paypal.com.evil.cn/x' -> 'evil.cn'; IP addresses are returned as-is."""
    host = (_parts(url).hostname or "").lower()
    try:
        ipaddress.ip_address(host)
        return host
    except ValueError:
        pass
    ext = _EXTRACT(host)
    return ".".join(p for p in (ext.domain, ext.suffix) if p) or host


def _entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    return -sum(c / len(text) * math.log2(c / len(text)) for c in counts.values())


def _looks_like_brand(domain: str) -> bool:
    """True for 'paypa1', 'paypal-secure', 'g00gle-login'; False for 'paypal' itself.

    Short brands (irs, dhl, chase, apple...) must match a whole hyphen-separated
    piece, so 'first' or 'purchase' don't count as IRS or Chase.
    """
    squashed = domain.translate(_LOOKALIKE)
    tokens = squashed.split("-")
    joined = squashed.replace("-", "")
    for b in BRANDS:
        if domain == b:
            continue
        if len(b) <= 5:
            if b in tokens:
                return True
        elif b in joined:
            return True
    return False


def lexical_features(url: str) -> list[float]:
    p = _parts(url)
    host = (p.hostname or "").lower()
    ext = _EXTRACT(host)
    try:
        ipaddress.ip_address(host)
        is_ip = 1.0
    except ValueError:
        is_ip = 0.0
    sub = ext.subdomain.lower()
    if sub.startswith("www.") or sub == "www":
        sub = sub[4:]
    domain = ext.domain.lower()
    rest = (sub + " " + p.path + " " + (p.query or "")).lower()

    rest_tokens = set(re.split(r"[^a-z0-9]+", rest))
    brand_elsewhere = any(
        b != domain and (b in rest_tokens if len(b) <= 5 else b in rest) for b in BRANDS
    )
    lookalike = _looks_like_brand(domain)
    lure = sum(w in (sub + p.path).lower() for w in LURE_WORDS)
    try:
        has_port = 1.0 if p.port else 0.0
    except ValueError:
        has_port = 1.0

    return [
        len(host),
        len(p.path),
        len([q for q in (p.query or "").split("&") if q]),
        len([s for s in sub.split(".") if s]),
        host.count("."),
        host.count("-"),
        sum(ch.isdigit() for ch in host),
        _entropy(host),
        is_ip,
        1.0 if "@" in url else 0.0,
        1.0 if "xn--" in host else 0.0,
        1.0 if ext.suffix.split(".")[-1] in RISKY_TLDS else 0.0,
        1.0 if brand_elsewhere else 0.0,
        1.0 if lookalike else 0.0,
        float(lure),
        has_port,
        1.0 if "//" in p.path else 0.0,
        len(normalize(url)),
    ]


def lexical_matrix(urls) -> np.ndarray:
    return np.array([lexical_features(u) for u in urls], dtype=float)


def explain(url: str) -> list[str]:
    """Plain-English reasons from the hand-built features, for API responses."""
    f = dict(zip(FEATURE_NAMES, lexical_features(url)))
    reasons = []
    if f["is_ip"]:
        reasons.append("Uses a raw IP address instead of a domain name")
    if f["lookalike_brand"]:
        reasons.append("Domain imitates a well-known brand name but isn't that brand's site")
    if f["brand_not_registered_domain"]:
        reasons.append("Mentions a well-known brand outside the site's real domain")
    if f["has_punycode"]:
        reasons.append("Uses punycode (xn--), which can hide lookalike characters")
    if f["has_at"]:
        reasons.append("Contains '@', which can hide the real destination")
    if f["risky_tld"]:
        reasons.append("Uses a domain ending often seen in abuse")
    if f["lure_words"] >= 2:
        reasons.append("Contains several login/verify-style words")
    if f["subdomain_count"] >= 3:
        reasons.append("Has an unusually deep chain of subdomains")
    return reasons


ML_DIR = Path(__file__).resolve().parent
MODEL_PATH = ML_DIR / "phishing_model.pkl"


def load_url_model(path=MODEL_PATH):
    """Load the saved URL pipeline from any working directory."""
    import sys

    if str(ML_DIR) not in sys.path:
        sys.path.insert(0, str(ML_DIR))
    return joblib.load(path)


def phishing_probability(model, url: str) -> float:
    """Probability (0-1) that the URL is phishing."""
    proba = model.predict_proba([url])[0]
    return float(proba[list(model.classes_).index(1)])


TOP_DOMAINS_PATH = ML_DIR / "data" / "top_domains.txt"
_top_domains: set[str] | None = None


def _load_top_domains() -> set[str]:
    global _top_domains
    if _top_domains is None:
        if TOP_DOMAINS_PATH.exists():
            lines = TOP_DOMAINS_PATH.read_text().splitlines()
            _top_domains = {ln.strip() for ln in lines if ln.strip() and not ln.startswith("#")}
        else:
            _top_domains = set()  # list not built yet: every URL goes to the model
    return _top_domains


def known_domain(url: str) -> bool:
    """True if the URL's real domain is a well-known site (Tranco top 10k)
    and not a place where anyone can host a page."""
    host = (_parts(url).hostname or "").lower()
    if any(host == h or host.endswith("." + h) for h in SHARED_HOSTS):
        return False
    reg = registered_domain(url)
    return reg not in SHARED_HOSTING and reg in _load_top_domains()


KNOWN_DOMAIN_REASON = "Well-known domain (Tranco top 10,000), checked before the model"


def score_url(model, url: str) -> tuple[float, list[str]]:
    """Phishing probability and reasons: the known-domain check first, then the model."""
    if known_domain(url):
        return 0.0, [KNOWN_DOMAIN_REASON]
    return phishing_probability(model, url), explain(url)
