import joblib
from preprocess import load_and_preprocess
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

import xgboost as xgb



X_train, X_test, y_train, y_test, vectorizer = load_and_preprocess("ml/data/final_dataset.csv")

lr_model = LogisticRegression(max_iter=1000)
lr_model.fit(X_train, y_train)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    eval_metric="logloss",
    random_state=42,
)

xgb_model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    verbose=50,
)

joblib.dump(rf_model, "ml/phishing_model.pkl")
joblib.dump(vectorizer, "ml/vectorizer.pkl")
joblib.dump(xgb_model, "ml/phishing_model_xgb.pkl")

print("Model training completed successfully")