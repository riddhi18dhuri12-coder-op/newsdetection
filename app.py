"""Flask web application for Fake News Detection."""

import os
import re

import joblib
from flask import Flask, jsonify, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "fake_news_model.pkl")
VECTORIZER_PATH = os.path.join(BASE_DIR, "model", "tfidf_vectorizer.pkl")

MAX_TEXT_LENGTH = 10000

app = Flask(__name__)

# Load the pre-trained artifacts once at startup. The model is NOT retrained here.
try:
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
except FileNotFoundError:
    model = None
    vectorizer = None
    print("WARNING: model files not found. Run `python train_model.py` first.")


def clean_text(text):
    """Must match the cleaning used in train_model.py."""
    text = str(text).lower()
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if model is None or vectorizer is None:
        return jsonify({"error": "Model files are missing. Run 'python train_model.py' and restart the server."}), 503

    data = request.get_json(silent=True) or {}
    text = data.get("text", "")

    if not isinstance(text, str) or not text.strip():
        return jsonify({"error": "Please enter some news text to analyze."}), 400

    if len(text) > MAX_TEXT_LENGTH:
        return jsonify({"error": f"Text is too long. Please keep it under {MAX_TEXT_LENGTH} characters."}), 400

    cleaned = clean_text(text)
    if not cleaned:
        return jsonify({"error": "The text contains no readable words. Please enter a news headline or article."}), 400

    try:
        features = vectorizer.transform([cleaned])
        prediction = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]
        confidence = float(probabilities[list(model.classes_).index(prediction)]) * 100
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Something went wrong while analyzing the text. Please try again."}), 500

    return jsonify({"prediction": str(prediction), "confidence": round(confidence, 2)})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
