import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess(filepath: str):
    df = pd.read_csv(filepath)

    df.dropna(subset=["text", "label"], inplace=True)

    df["label"] = pd.to_numeric(df["label"])

    vectorizer = TfidfVectorizer(
        max_features=10000,
        analyzer="word",
        ngram_range=(1, 2),
        stop_words="english",
        min_df=2,
    )
    X = vectorizer.fit_transform(df["text"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, vectorizer
