"""Train the URL phishing model with an honest evaluation.

What changed from the first version:
  * Data: ml/data/final_dataset.csv (real safe URLs with paths), built by
    ml/data/build_dataset.py.
  * URLs are normalized first (no scheme, no leading 'www.'), so the model
    can't learn "https://www." = safe.
  * Train/test are split by registered domain, so no website appears in both.
  * The TF-IDF vocabulary is learned from the training split only.
  * Character n-grams are combined with hand-built features (IP host,
    lookalike brand, brand outside the real domain, punycode, risky TLD...).
  * Results come from 5-fold cross-validation grouped by domain, plus a final
    held-out test split and a set of hand-picked hard cases.

Outputs:
  ml/phishing_model.pkl        the full pipeline: raw URL in, prediction out
                               (replaces the old model + vectorizer.pkl pair)
  ml/model_metrics.json        the numbers to put in the README
  confusion_matrix.png         held-out test results

Run from anywhere:  python3 phishing-detector/ml/train.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, MaxAbsScaler
from xgboost import XGBClassifier

ML_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ML_DIR))
from preprocess import MODEL_PATH, lexical_matrix, normalize_many, registered_domain  # noqa: E402

DATA = ML_DIR / "data" / "final_dataset.csv"
MODEL_OUT = MODEL_PATH
METRICS_OUT = ML_DIR / "model_metrics.json"
PLOT_OUT = ML_DIR.parent / "confusion_matrix.png"
SEED = 42

# URLs the first model got wrong, plus a few classic phishing patterns.
# 0 = legitimate, 1 = phishing
HARD_CASES = [
    ("https://github.com/by-tayo/ai-phishing-detector", 0),
    ("https://www.amazon.com/gp/your-account/order-history", 0),
    ("https://docs.python.org/3/library/email.html", 0),
    ("https://www.paypal.com/signin", 0),
    ("https://accounts.google.com", 0),
    ("https://www.utsa.edu/", 0),
    ("https://www.paypa1-secure-login.com", 1),
    ("https://www.chase-verify-account.com", 1),
    ("http://apple.com.aieuv.cn/mim/verify/index.html", 1),
    ("http://192.168.1.1/login", 1),
    ("https://secure-update-billing.xyz/netflix/login.php", 1),
]


def build_pipeline(classifier) -> Pipeline:
    features = FeatureUnion([
        ("chars", Pipeline([
            ("normalize", FunctionTransformer(normalize_many)),
            ("tfidf", TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                                      max_features=20000, sublinear_tf=True)),
        ])),
        ("lexical", Pipeline([
            ("extract", FunctionTransformer(lexical_matrix)),
            ("scale", MaxAbsScaler()),
        ])),
    ])
    return Pipeline([("features", features), ("model", classifier)])


CANDIDATES = {
    "logistic_regression": lambda: LogisticRegression(max_iter=3000, C=4.0),
    "xgboost": lambda: XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                                     subsample=0.9, colsample_bytree=0.5,
                                     eval_metric="logloss", random_state=SEED, n_jobs=-1),
}


def scores(y_true, y_pred) -> dict:
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
    }


def save_confusion_matrix(cm, model_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")  # no window needed (works in Codespaces and Docker)
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(cm, cmap="Blues")
    labels = ["Legitimate", "Phishing"]
    ax.set_xticks([0, 1], labels)
    ax.set_yticks([0, 1], labels)
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i][j]:,}", ha="center", va="center",
                    color="white" if cm[i][j] > max(map(max, cm)) / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"URL model ({model_name}) on unseen websites")
    fig.tight_layout()
    fig.savefig(PLOT_OUT, dpi=150)
    plt.close(fig)


def main() -> None:
    if not DATA.exists():
        sys.exit(f"Missing {DATA}. Run ml/data/build_dataset.py first.")

    df = pd.read_csv(DATA).dropna(subset=["url", "label"])
    df["group"] = df["url"].map(registered_domain)
    X, y, groups = df["url"].to_numpy(), df["label"].astype(int).to_numpy(), df["group"].to_numpy()
    print(f"{len(df):,} URLs from {df['group'].nunique():,} different websites\n")

    # 1) Hold out 20% of websites for the final test.
    train_idx, test_idx = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=SEED)
                               .split(X, y, groups))
    shared = set(groups[train_idx]) & set(groups[test_idx])
    assert not shared, f"websites in both train and test: {list(shared)[:5]}"

    # 2) Pick the model with 5-fold cross-validation on the training websites only.
    cv_results = {}
    for name, make in CANDIDATES.items():
        fold_f1 = []
        for tr, va in GroupKFold(n_splits=5).split(X[train_idx], y[train_idx], groups[train_idx]):
            pipe = build_pipeline(make())
            pipe.fit(X[train_idx][tr], y[train_idx][tr])
            fold_f1.append(f1_score(y[train_idx][va], pipe.predict(X[train_idx][va])))
        cv_results[name] = {"f1_mean": round(float(np.mean(fold_f1)), 4),
                            "f1_std": round(float(np.std(fold_f1)), 4)}
        print(f"5-fold CV  {name:<20} F1 {np.mean(fold_f1):.3f} ± {np.std(fold_f1):.3f}")
    best = max(cv_results, key=lambda n: cv_results[n]["f1_mean"])
    print(f"\nBest: {best}")

    # 3) Fit the best model on all training websites; score the held-out websites once.
    pipe = build_pipeline(CANDIDATES[best]())
    pipe.fit(X[train_idx], y[train_idx])
    pred = pipe.predict(X[test_idx])
    test_scores = scores(y[test_idx], pred)
    tn, fp, fn, tp = confusion_matrix(y[test_idx], pred, labels=[0, 1]).ravel()
    print(f"\nHeld-out test ({len(test_idx):,} URLs from websites never seen in training):")
    for k, v in test_scores.items():
        print(f"  {k:<10} {v:.2%}")
    print(f"  safe URLs wrongly flagged: {fp}   phishing URLs missed: {fn}")
    save_confusion_matrix([[tn, fp], [fn, tp]], best)

    # 4) Hand-picked hard cases.
    print("\nHard cases (✓ = correct):")
    hard_ok = 0
    for url, want in HARD_CASES:
        p = float(pipe.predict_proba([url])[0][1])
        ok = (p >= 0.5) == bool(want)
        hard_ok += ok
        print(f"  {'✓' if ok else '✗'} {'PHISH' if p >= 0.5 else 'safe ':5} {p:5.1%}  {url}")
    print(f"  {hard_ok}/{len(HARD_CASES)} correct")

    # 5) Final model: refit on all data so the API gets every example.
    final = build_pipeline(CANDIDATES[best]())
    final.fit(X, y)
    joblib.dump(final, MODEL_OUT)

    metrics = {
        "dataset": "Hannousse & Yahiouche (2021) via Hugging Face pirocheto/phishing-url (CC BY 4.0)",
        "urls": int(len(df)), "websites": int(df["group"].nunique()),
        "split": "grouped by registered domain; no website in both train and test",
        "model": best, "cross_validation": cv_results,
        "held_out_test": {**test_scores, "test_urls": int(len(test_idx)),
                          "false_positives": int(fp), "false_negatives": int(fn)},
        "hard_cases_correct": f"{hard_ok}/{len(HARD_CASES)}",
    }
    METRICS_OUT.write_text(json.dumps(metrics, indent=2))
    print(f"\nSaved {MODEL_OUT.name} and {METRICS_OUT.name} in {ML_DIR}, and {PLOT_OUT.name}")


if __name__ == "__main__":
    main()
