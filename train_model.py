"""
Train the Fake News Detection model.

Usage:
    python train_model.py

Reads  : data/news.csv
Writes : model/fake_news_model.pkl
         model/tfidf_vectorizer.pkl
"""

import os
import re
import sys

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "news.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(MODEL_DIR, "tfidf_vectorizer.pkl")

RANDOM_STATE = 42
TEST_SIZE = 0.25


def clean_text(text):
    """Lowercase the text, remove unnecessary punctuation and extra spaces."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)  # remove URLs
    text = re.sub(r"[^a-z0-9\s]", " ", text)            # remove punctuation/symbols
    text = re.sub(r"\s+", " ", text).strip()            # collapse whitespace
    return text


def load_dataset(path):
    """Load the CSV, validate its columns and drop unusable rows."""
    if not os.path.exists(path):
        sys.exit(f"Error: dataset not found at {path}")

    df = pd.read_csv(path)

    missing_columns = {"text", "label"} - set(df.columns)
    if missing_columns:
        sys.exit(f"Error: missing required column(s): {', '.join(sorted(missing_columns))}")

    # Handle missing values
    df = df.dropna(subset=["text", "label"]).copy()
    df["label"] = df["label"].astype(str).str.strip().str.upper()
    df = df[df["label"].isin(["FAKE", "REAL"])]

    if df.empty:
        sys.exit("Error: no valid rows found. Labels must be FAKE or REAL.")
    if df["label"].nunique() < 2:
        sys.exit("Error: the dataset needs both FAKE and REAL examples.")

    df["clean_text"] = df["text"].apply(clean_text)
    df = df[df["clean_text"] != ""]
    return df


def main():
    df = load_dataset(DATA_PATH)
    print(f"Loaded {len(df)} records")
    print(df["label"].value_counts().to_string(), "\n")

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"],
        df["label"],
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=df["label"],
    )

    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=5000)
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)

    model = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    model.fit(X_train_tfidf, y_train)

    predictions = model.predict(X_test_tfidf)

    # FAKE is treated as the positive class for precision / recall / F1
    print("Evaluation on the held-out test set")
    print("-----------------------------------")
    print(f"Training samples : {len(X_train)}")
    print(f"Testing samples  : {len(X_test)}")
    print(f"Accuracy         : {accuracy_score(y_test, predictions):.4f}")
    print(f"Precision        : {precision_score(y_test, predictions, pos_label='FAKE', zero_division=0):.4f}")
    print(f"Recall           : {recall_score(y_test, predictions, pos_label='FAKE', zero_division=0):.4f}")
    print(f"F1-score         : {f1_score(y_test, predictions, pos_label='FAKE', zero_division=0):.4f}")
    print("\nNote: this is a tiny demo dataset. These numbers are NOT a reliable")
    print("measure of real-world fake news detection performance.\n")

    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    print(f"Saved model      -> {os.path.relpath(MODEL_PATH, BASE_DIR)}")
    print(f"Saved vectorizer -> {os.path.relpath(VECTORIZER_PATH, BASE_DIR)}")


if __name__ == "__main__":
    main()
