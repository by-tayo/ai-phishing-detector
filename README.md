
# 🛡️ AI-Powered Phishing Detector

## 📌 Overview

This project builds a **full-stack AI-powered phishing detection system** that classifies both **URLs** and **emails** as phishing or legitimate in real time.
The pipeline:

* Downloads and preprocesses phishing URLs from **PhishTank** and legitimate URLs from **Majestic Million**.
* Trains **three ML models** — Logistic Regression, Random Forest, and XGBoost — using TF-IDF character n-gram features.
* Fine-tunes a **DistilBERT transformer** model for deep contextual URL understanding.
* Trains a dedicated **email text classifier** (TF-IDF + Logistic Regression/Random Forest/XGBoost) on a labeled phishing/legitimate email corpus.
* Runs rule-based **email header and content heuristics** — sender/reply-to mismatches, brand impersonation, urgency language, mismatched links, IP-based links — and reuses the URL model on any links found in an email.
* Exposes predictions via a **FastAPI REST API** with five endpoints.
* Displays results in an interactive **Plotly Dash frontend** with a URL Scanner and an Email Scanner tab, each with red/green result cards.
* Provisions cloud infrastructure on **AWS** using **Terraform**.

---

## 📑 Table of Contents

1. [Project Objectives](#-project-objectives)
2. [Project Structure](#-project-structure)
3. [Dataset](#-dataset)
4. [ML Models & Results](#-ml-models--results)
5. [API Endpoints](#-api-endpoints)
6. [Frontend](#-frontend)
7. [AWS Infrastructure](#-aws-infrastructure)
8. [How to Run](#-how-to-run)
9. [Tools and Libraries](#-tools-and-libraries)
10. [Example Predictions](#-example-predictions)
11. [Possible Extensions](#-possible-extensions)

---

## 🎯 Project Objectives

* Build a real-time phishing URL classifier using multiple AI approaches.
* Engineer character-level TF-IDF features that capture phishing URL patterns.
* Fine-tune a pre-trained DistilBERT transformer on a balanced phishing dataset.
* Build a real-time phishing **email** classifier combining an ML text model, header/content heuristics, and link analysis.
* Expose predictions through a documented REST API built with FastAPI.
* Build an interactive frontend dashboard using Plotly Dash.
* Provision reproducible cloud infrastructure on AWS using Terraform IaC.
* Integrate a live threat feed (OpenPhish) for real-time phishing verification.

---

## 📂 Project Structure

*(Note: Large datasets, `.csv` files, heavy model weights, and local `.terraform` states are explicitly `.gitignored` to keep the repository lightweight.)*

```text
phishing-detector/
├── backend/
│   ├── infra/                       # Terraform AWS configuration
│   │   ├── .terraform.lock.hcl      # Terraform dependency lock
│   │   ├── main.tf                  # Core AWS resource definitions
│   │   ├── outputs.tf               # Resource URLs and identifiers
│   │   ├── provider.tf              # AWS provider configuration
│   │   └── variables.tf             # Input variables
│   ├── main.py                      # FastAPI application
│   ├── model.py                     # Backend ML model loader
│   ├── requirements.txt             # Backend dependencies
│   └── schemas.py                   # Request/response schemas
├── ml/
│   ├── data/
│   │   ├── build_dataset.py         # URL dataset construction script
│   │   ├── combine_dataset.py       # URL dataset merging script (generates local CSVs)
│   │   └── build_email_dataset.py   # Email dataset construction script (Hugging Face)
│   ├── models/
│   │   └── phishing-bert/           # DistilBERT configuration files
│   │       ├── config.json
│   │       ├── tokenizer.json
│   │       └── tokenizer_config.json
│   ├── my_dash_app_frontend/        # Plotly Dash frontend application
│   │   ├── app.py                   # Frontend main application (URL + Email tabs)
│   │   └── assets/
│   │       └── style.css            # Custom CSS styling
│   ├── evaluate.py                  # Scikit-learn URL model evaluation
│   ├── evaluate_email.py            # Email text model evaluation
│   ├── evaluate_transformer.py      # Transformer model evaluation
│   ├── fine_tune.py                 # DistilBERT fine-tuning script
│   ├── model.py                     # ML model architecture/definitions
│   ├── email_heuristics.py          # Email header/content heuristic analysis
│   ├── email_phishing_model.pkl     # Saved email text Random Forest model
│   ├── email_phishing_model_xgb.pkl # Saved email text XGBoost model
│   ├── email_vectorizer.pkl         # Saved email TF-IDF vectorizer
│   ├── phishing_model.pkl           # Saved URL Random Forest model
│   ├── phishing_model_xgb.pkl       # Saved URL XGBoost model
│   ├── phishtank_api.py             # OpenPhish live feed integration
│   ├── predict.py                   # URL ML model inference
│   ├── predict_email.py             # Combined email prediction (text model + heuristics + link scoring)
│   ├── predict_transformer.py       # Transformer inference
│   ├── preprocess.py                # URL TF-IDF preprocessing pipeline
│   ├── preprocess_email.py          # Email TF-IDF preprocessing pipeline
│   ├── preprocess_phishtank.py      # PhishTank data cleaning script
│   ├── tokenize_data.py             # Hugging Face tokenization
│   ├── train.py                     # URL model training (LR, RF, XGBoost)
│   ├── train_email.py               # Email model training (LR, RF, XGBoost)
│   └── vectorizer.pkl               # Saved URL TF-IDF vectorizer
├── confusion_matrix.png             # ML model evaluation plot
├── confusion_matrix_bert.png        # BERT evaluation plot
├── confusion_matrix_email_text.png  # Email text model evaluation plot
├── Dockerfile                       # Docker container configuration
├── dockerignore                     # Docker ignore rules
├── .gitignore                       # Git ignore rules
└── README.md
```

---

## 📊 Dataset

### Phishing URLs — PhishTank
* Source: [PhishTank](http://data.phishtank.com/data/online-valid.csv) — operated by Cisco Talos Intelligence Group.
* Filtered to verified phishing entries only (`verified == 'yes'`).
* Key columns used: `url`, `verified`, `target`.
* **55,877 verified phishing URLs** after preprocessing.

### Legitimate URLs — Majestic Million
* Source: [Majestic Million](https://majestic.com/reports/majestic-million) — top 1 million most-visited domains.
* Formatted as `https://www.[domain]` and labeled as legitimate.
* **55,877 legitimate URLs sampled** to match phishing count.

### Combined Dataset
* **111,754 total URLs** — perfectly balanced (50/50 split).
* Labels: `1 = phishing`, `0 = legitimate`.
* Saved to `ml/data/final_dataset.csv`.

### Emails — Phishing Email Dataset (Hugging Face)
* Source: [`zefang-liu/phishing-email-dataset`](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) — a combined Enron (legitimate) and Nazario (phishing) email corpus.
* **18,650 raw labeled emails** (`Email Text` / `Email Type`) → **17,537 after dropping nulls and duplicates** (10,979 legitimate / 6,558 phishing).
* Labels: `1 = phishing`, `0 = legitimate` (safe).
* Saved to `ml/data/email_dataset.csv`.

---

## 🤖 ML Models & Results

### Feature Engineering — TF-IDF
URL text was converted to numerical features using TF-IDF with character n-grams:

* `max_features`: 5,000
* `analyzer`: `char_wb` (character-level within word boundaries)
* `ngram_range`: `(2, 4)` — captures patterns like `login`, `verify`, `secure`

### Scikit-learn + XGBoost Models

| Model | Accuracy | Precision | Recall | F1 Score |
|-------|----------|-----------|--------|----------|
| Logistic Regression | ~99% | ~99% | ~99% | ~99% |
| Random Forest | 99.92% | 99.98% | 99.87% | 99.92% |
| XGBoost | 99.92% | 99.98% | 99.87% | 99.92% |

XGBoost validation log-loss dropped from `0.599` at round 0 to `0.015` by round 50, stabilizing at `0.014` at completion.

### Email Text Classifier (TF-IDF + Random Forest)

Email body text was converted to features using word-level TF-IDF (`max_features=10000`, `ngram_range=(1,2)`, English stop words removed) and trained on the 17,537-email dataset above:

| Metric | Score |
|--------|-------|
| Accuracy | 97.21% |
| Precision | 95.85% |
| Recall | 96.73% |
| F1 Score | 96.29% |

XGBoost validation log-loss on the same split dropped from `0.621` at round 0 to `0.113` by round 199. The Random Forest model is used as the primary email text classifier in `/predict/email`, combined with header/content heuristics and URL-model scoring of any links found in the email (see [API Endpoints](#-api-endpoints)).

### DistilBERT Transformer (Fine-tuned on Google Colab T4 GPU)

| Epoch | Training Loss | Validation Loss | Accuracy | F1 Score |
|-------|--------------|-----------------|----------|----------|
| 1 | 0.004179 | 0.020120 | 99.75% | 99.75% |
| 2 | 0.003817 | 0.003067 | 99.96% | 99.96% |
| 3 | 0.001172 | 0.002017 | **99.98%** | **99.98%** |

The fine-tuned model achieved **perfect 100% accuracy** on a 500-sample evaluation with zero false positives and zero false negatives.

### Saved Model Artifacts

* `ml/phishing_model.pkl` — URL Random Forest model (4.15 MB)
* `ml/phishing_model_xgb.pkl` — URL XGBoost model (282 KB)
* `ml/vectorizer.pkl` — URL TF-IDF vectorizer (165 KB)
* `ml/models/phishing-bert/` — Fine-tuned DistilBERT model folder
* `ml/email_phishing_model.pkl` — Email text Random Forest model (~41 MB)
* `ml/email_phishing_model_xgb.pkl` — Email text XGBoost model (368 KB)
* `ml/email_vectorizer.pkl` — Email TF-IDF vectorizer (373 KB)

---

## 🔌 API Endpoints

The FastAPI backend exposes five prediction endpoints:

### `POST /predict`
XGBoost / Random Forest ML model prediction.

```json
// Request
{ "text": "http://suspicious-login.verify-account.com" }

// Response
{ "input": "http://suspicious-login.verify-account.com", "is_phishing": true, "confidence": 1.0 }
```

### `POST /predict/transformer`
Fine-tuned DistilBERT transformer prediction.

```json
// Request
{ "text": "https://www.google.com" }

// Response
{ "input": "https://www.google.com", "is_phishing": false, "confidence": 0.9999 }
```

### `POST /check-live`
Real-time OpenPhish live threat database lookup.

```json
// Request
{ "text": "http://suspicious-login.com" }

// Response
{ "url": "http://suspicious-login.com", "in_database": false, "verified": false, "source": "OpenPhish" }
```

### `POST /predict/email`
Combined email phishing prediction — merges the email text classifier, sender/header heuristics, and URL-model scoring of any links found in the email body. Accepts a raw RFC-822 email (headers + body, e.g. pasted from "Show Original") or just plain body text.

```json
// Request
{ "raw_email": "From: PayPal Support <security@paypa1-verify.com>\nReply-To: help@another-domain.net\nSubject: Urgent: Verify your account\n\nDear Customer, your account has been suspended due to unusual activity. Please verify your account immediately: http://192.168.1.1/login" }

// Response
{
  "is_phishing": true,
  "confidence": 0.9626,
  "text_model_confidence": 0.9253,
  "heuristic_score": 1,
  "reasons": [
    "Reply-To domain (another-domain.net) differs from From domain (paypa1-verify.com)",
    "Sender name mentions 'Paypal' but the address domain is 'paypa1-verify.com', not an official Paypal domain",
    "Urgent/pressure language detected: verify your account, account has been suspended, unusual activity",
    "Generic greeting used instead of a personalized name",
    "Link points directly to an IP address instead of a domain"
  ],
  "links": [
    { "url": "http://192.168.1.1/login", "is_phishing": true, "confidence": 1 }
  ],
  "sender": "security@paypa1-verify.com",
  "subject": "Urgent: Verify your account"
}
```

Interactive API documentation is auto-generated by FastAPI.

---

## 🖥️ Frontend

The Plotly Dash frontend (`my_dash_app_frontend/app.py`) provides two tabs:

**🔗 URL Scanner**
* **URL input field** — enter any URL to check
* **Model selector dropdown** — choose between XGBoost, DistilBERT, or Live Check
* **⚠️ Red card** — displayed for phishing URLs with confidence percentage
* **✅ Green card** — displayed for legitimate URLs with confidence percentage

**📧 Email Scanner**
* **Textarea** — paste a raw email (headers + body) or just the body text
* **Combined verdict card** — red/green card with overall confidence
* **Reasons list** — the specific heuristic flags that were triggered (sender/reply-to mismatch, brand impersonation, urgency language, mismatched or IP-based links, etc.)
* **Per-link results** — every URL found in the email, individually scored by the URL model

Both tabs share:
* **Loading spinner** — shown while waiting for API response
* **How it works section** — explains all detection methods

---

## ☁️ AWS Infrastructure

All AWS resources were provisioned using **Terraform** (Infrastructure as Code):

| AWS Resource | Details |
|-------------|---------|
| IAM Role | Permissions and access management |
| VPC | Isolated network |
| S3 Bucket | ML model artifact and log storage |
| RDS MySQL 8.0 | Phishing scan history database (`db.t3.micro`, 20 GB) |

### Terraform Commands

```bash
cd backend/infra
terraform init      # Initialize providers
terraform plan      # Preview changes
terraform apply     # Deploy infrastructure
terraform destroy   # Tear down infrastructure
```

---

## ⚙️ How to Run

### Prerequisites
* Python 3.12+
* Cursor IDE
* WSL (Ubuntu) or Linux
* AWS CLI configured (`aws configure`)
* Terraform installed

### 1. Activate virtual environment
```bash
source venv/bin/activate
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Build the dataset(s)
```bash
python3 ml/preprocess_phishtank.py
python3 ml/data/combine_dataset.py
python3 ml/data/build_email_dataset.py
```

### 4. Train the models
```bash
python3 ml/train.py
python3 ml/train_email.py
```

### 5. Start the FastAPI backend
```bash
cd backend
uvicorn main:app --reload
```
* Swagger docs: `http://localhost:8000/docs`

### 6. Start the Dash frontend
Open a new terminal:
```bash
python3 my_dash_app_frontend/app.py
```

* Dashboard: `http://localhost:8050`

---

## 🛠 Tools and Libraries

* **Python 3.12** — core language
* **Scikit-learn** — Logistic Regression, Random Forest, TF-IDF, evaluation metrics
* **XGBoost** — gradient boosting classifier
* **Hugging Face Transformers** — DistilBERT fine-tuning and inference
* **Hugging Face Datasets** — downloading the phishing email corpus
* **PyTorch** — deep learning backend for transformer training
* **FastAPI** — REST API framework
* **Uvicorn** — ASGI server
* **Plotly Dash** — interactive frontend dashboard
* **Dash Bootstrap Components** — UI styling
* **Pandas / NumPy** — data processing
* **Matplotlib / Seaborn** — confusion matrix and evaluation plots
* **Joblib** — model serialization
* **Requests** — HTTP calls to OpenPhish
* **Terraform** — AWS infrastructure as code
* **AWS CLI** — cloud credential management
* **Google Colab** — GPU training for DistilBERT (T4 GPU)
* **Cursor IDE** — development environment
* **WSL (Ubuntu)** — Linux environment on Windows

---

## 🧪 Example Predictions

| URL | Result | Confidence |
|-----|--------|-----------|
| `http://suspicious-login.verify-account.com` | ⚠️ Phishing | 100.00% |
| `http://secure-login.verify-account.phishing.com` | ⚠️ Phishing | 100.00% |
| `https://www.google.com` | ✅ Safe | 99.99% |
| `https://www.amazon.com` | ✅ Safe | 100.00% |
| `https://www.github.com` | ✅ Safe | 100.00% |

**Email examples** (via `/predict/email`, real output from this build):

| Email | Result | Confidence | Key reasons |
|-------|--------|-----------|--------------|
| Spoofed "PayPal Support" sender, urgent tone, IP-based link | ⚠️ Phishing | 96.26% | Reply-To/From mismatch, brand impersonation, urgency language, IP link |
| "Hi team, checking in about tomorrow's 3pm meeting..." | ✅ Safe | 1.25% | No flags triggered |

---

## 🚀 Possible Extensions

* Integrate the RDS MySQL database to log all prediction history and build a scan dashboard.
* Fine-tune a DistilBERT transformer on email text (the current email classifier is TF-IDF + ML, mirroring the non-transformer URL pipeline) for deeper contextual understanding.
* Deploy the FastAPI backend to AWS using EC2 or App Runner.
* Build a React/Next.js frontend for a more polished user interface.
* Add the PhishTank live API integration when registration reopens.
* Implement rate limiting on the FastAPI endpoints to prevent abuse.
* Add GitHub Actions CI/CD for automated testing on every push.
* Train on a larger dataset combining multiple phishing sources.
