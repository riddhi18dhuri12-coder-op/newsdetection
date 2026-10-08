# Fake News Detection using NLP and Machine Learning

A beginner-friendly web application that classifies a pasted news headline or article as **REAL** or **FAKE** using a real NLP/ML pipeline (text cleaning, TF-IDF and Logistic Regression). The trained model is saved with `joblib` and loaded by a Flask app, so nothing is hardcoded and nothing is retrained when the server starts.

> **Disclaimer:** This project is an educational NLP demonstration. The prediction is based on patterns learned from the training dataset and should not be treated as a definitive fact-checking system.

> **Demo dataset:** `data/news.csv` contains only about 70 short sample sentences so the project runs immediately. It is **not** enough to build a reliable detector. A real project should use a much larger, verified dataset (thousands of labelled articles). See [Limitations](#limitations).

---

## Features

- Paste a headline or article and get a REAL / FAKE prediction with a confidence percentage
- Real ML pipeline: cleaning, TF-IDF vectorization, Logistic Regression
- Training script with validation, missing-value handling and evaluation metrics
- Model saved with `joblib` and loaded once at startup
- Clean, responsive interface (HTML, CSS, vanilla JavaScript) with loading and error states
- JSON API endpoint (`POST /predict`)
- Ready for Render (`gunicorn`, `PORT` environment variable, relative paths)
- No database, no API keys, no paid services

## Tech Stack

| Layer | Technology |
|-------|------------|
| Language | Python 3.11+ |
| Backend | Flask, gunicorn |
| ML / NLP | scikit-learn (TfidfVectorizer, LogisticRegression), pandas, numpy |
| Persistence | joblib (`.pkl` files) |
| Frontend | HTML5, CSS3, vanilla JavaScript (`fetch`) |

## NLP Pipeline

```
News Text
   ↓
Text Cleaning
   ↓
TF-IDF Vectorization
   ↓
Logistic Regression
   ↓
Prediction
   ↓
REAL / FAKE
```

1. **Text cleaning** – lowercase the text, remove URLs, remove punctuation and symbols, collapse extra whitespace. The same `clean_text()` logic is used in training and in the web app so the model always sees consistent input.
2. **TF-IDF vectorization** – converts text to numbers. Words (and word pairs) that are frequent in a document but rare across the dataset get higher weights. English stop words are removed.
3. **Logistic Regression** – a linear classifier that learns which weighted terms point towards FAKE or REAL and outputs a probability for each class.
4. **Prediction** – the class with the higher probability is returned together with its probability as the confidence percentage.

## Machine Learning Algorithm

**Logistic Regression** is a simple, fast and interpretable binary classifier. It computes a weighted sum of the TF-IDF features and passes it through a sigmoid function to produce a probability between 0 and 1. It is a standard baseline for text classification and works well for a beginner project.

The training script splits the data 75% / 25% (train / test) with `random_state=42` (stratified, so both classes appear in both splits) and reports accuracy, precision, recall and F1-score. FAKE is treated as the positive class.

## Project Structure

```
fake-news-detector/
│
├── app.py                  # Flask app (loads model, serves UI and /predict)
├── train_model.py          # Trains and saves the model
├── requirements.txt        # Python dependencies
├── Procfile                # Render / gunicorn start command
├── README.md
├── .gitignore
│
├── data/
│   └── news.csv            # Demo dataset (text,label)
│
├── model/
│   ├── fake_news_model.pkl     # Trained Logistic Regression model
│   └── tfidf_vectorizer.pkl    # Fitted TF-IDF vectorizer
│
├── templates/
│   └── index.html
│
└── static/
    ├── style.css
    └── script.js
```

## Installation

Requires Python 3.11 or newer.

```bash
git clone https://github.com/<your-username>/fake-news-detector.git
cd fake-news-detector

python -m venv venv

# macOS / Linux
source venv/bin/activate
# Windows (PowerShell)
venv\Scripts\Activate.ps1
# Windows (cmd)
venv\Scripts\activate.bat

pip install -r requirements.txt
```

## How to Train the Model

```bash
python train_model.py
```

This reads `data/news.csv`, prints the evaluation metrics, and writes:

- `model/fake_news_model.pkl`
- `model/tfidf_vectorizer.pkl`

Re-run this command whenever you change the dataset. To use your own data, replace `data/news.csv` with a CSV that has the columns `text` and `label` (labels must be `FAKE` or `REAL`).

> **Important:** the model files must be created with the same scikit-learn version that runs the app. The versions in `requirements.txt` are pinned for this reason. If you upgrade a package, retrain the model.

## How to Run Locally

```bash
python app.py
```

Open http://localhost:5000 in your browser.

To test with the production server instead (macOS / Linux):

```bash
gunicorn app:app
```

## Render Deployment

1. **Create a GitHub repository** (empty, no README) at https://github.com/new.
2. **Upload the complete project** (all files and folders shown above).
3. **Make sure the trained model files exist.** Check that `model/fake_news_model.pkl` and `model/tfidf_vectorizer.pkl` are present. If not, see "If the `.pkl` files are missing" below.
4. **Push the project to GitHub:**
   ```bash
   git init
   git add .
   git commit -m "Initial commit: Fake News Detector"
   git branch -M main
   git remote add origin https://github.com/<your-username>/fake-news-detector.git
   git push -u origin main
   ```
5. **Open Render** at https://render.com and sign in (GitHub login is easiest).
6. Click **New +** and choose **Web Service**.
7. **Connect your GitHub repository** (`fake-news-detector`).
8. **Select the Python environment** (Runtime: `Python 3`). Choose the Free instance type for a demo.
9. **Build Command:**
   ```
   pip install -r requirements.txt
   ```
10. **Start Command:**
    ```
    gunicorn app:app
    ```
11. Under **Environment Variables**, add `PYTHON_VERSION` = `3.11.9` (keeps the Python version consistent with the pinned packages), then click **Create Web Service** to deploy.
12. When the deploy finishes, open the generated URL (for example `https://fake-news-detector.onrender.com`).

Note: free Render services sleep after inactivity, so the first request after a while can take 30-60 seconds.

### If the `.pkl` files are missing

If `model/*.pkl` were not committed (for example, they were excluded or lost), the app will start but `/predict` will return an error asking you to train the model. Fix it like this:

1. Run `python train_model.py` locally.
2. Verify the files exist in `model/`:
   ```bash
   ls model        # macOS / Linux
   dir model       # Windows
   ```
3. Commit them to GitHub:
   ```bash
   git add model/fake_news_model.pkl model/tfidf_vectorizer.pkl
   git commit -m "Add trained model files"
   git push
   ```
4. Render redeploys automatically (or click **Manual Deploy**).

## Example Predictions

Results come from a tiny demo dataset, so confidence values are modest and predictions on unfamiliar text can be wrong.

| Input | Expected output |
|-------|-----------------|
| Government announces new education policy for students | REAL |
| Health ministry launches vaccination drive for children in rural districts | REAL |
| City council approves new budget for road repairs | REAL |
| Shocking secret cure hidden by doctors, share this message now | FAKE |
| Aliens landed in the city and the government is hiding the truth | FAKE |
| Doctors hate this miracle trick that makes you rich overnight | FAKE |

API example:

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Government announces new education policy for students"}'
```

```json
{"prediction": "REAL", "confidence": 59.38}
```

## Limitations

- **Tiny demo dataset.** About 70 short sentences cannot teach a model what fake news looks like in the real world. The reported metrics are measured on a handful of test samples and are not meaningful as real-world performance.
- **Style, not truth.** The model learns word patterns in its training data. It does not check facts, sources or dates, so it can be confidently wrong.
- **Dataset bias.** The FAKE examples here are mostly sensational or absurd claims, so the model largely learns "sensational wording" rather than falsehood. Plausible-sounding fake news can be classified as REAL.
- **English only**, bag-of-words features (no deep understanding of context or sarcasm).
- **Confidence is not certainty.** The percentage is the model's probability estimate, not a guarantee.
- Not suitable for any real fact-checking or decision making.

## Future Improvements

- Train on a large verified dataset (for example the public "Fake and Real News" or LIAR datasets)
- Compare more models (Naive Bayes, Linear SVM, Random Forest) and use cross-validation
- Add lemmatization and better stop-word handling with NLTK or spaCy
- Show the most influential words behind each prediction (model explainability)
- Add a confusion matrix and evaluation dashboard
- Use transformer embeddings (BERT) for better context understanding
- Support multiple languages
- Add automated tests and a CI workflow
