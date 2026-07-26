import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

def load_and_preprocess(filepath: str):
    df = pd.read_csv(filepath)

    df.dropna(subset=["url", "label"], inplace=True)

    df["label"] = pd.to_numeric(df["label"])

    vectorizer = TfidfVectorizer(
        max_features=5000,
        analyzer="char_wb",
        ngram_range=(2,4),
    )
    X = vectorizer.fit_transform(df["url"])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, vectorizer