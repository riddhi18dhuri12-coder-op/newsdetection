const textInput = document.getElementById("news-text");
const analyzeBtn = document.getElementById("analyze-btn");
const clearBtn = document.getElementById("clear-btn");
const btnLabel = document.getElementById("btn-label");
const btnSpinner = document.getElementById("btn-spinner");
const errorBox = document.getElementById("error-message");
const resultCard = document.getElementById("result");
const badge = document.getElementById("result-badge");
const confidenceValue = document.getElementById("confidence-value");
const meterFill = document.getElementById("meter-fill");
const explanation = document.getElementById("explanation");

function setLoading(isLoading) {
  analyzeBtn.disabled = isLoading;
  btnSpinner.hidden = !isLoading;
  btnLabel.textContent = isLoading ? "Analyzing..." : "Analyze News";
}

function showError(message) {
  errorBox.textContent = message;
  errorBox.hidden = false;
  resultCard.hidden = true;
}

function showResult(prediction, confidence) {
  const label = prediction.toUpperCase();
  const cssClass = label === "REAL" ? "real" : "fake";

  badge.textContent = label;
  badge.className = "badge " + cssClass;
  confidenceValue.textContent = confidence.toFixed(2) + "%";
  meterFill.className = "meter-fill " + cssClass;
  meterFill.style.width = Math.min(confidence, 100) + "%";
  explanation.textContent =
    "The model classified this text as " + label + " based on learned linguistic patterns.";

  resultCard.hidden = false;
}

async function analyze() {
  const text = textInput.value.trim();
  errorBox.hidden = true;

  if (!text) {
    showError("Please paste a news article or headline first.");
    return;
  }

  setLoading(true);
  try {
    const response = await fetch("/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text }),
    });

    let data = {};
    try {
      data = await response.json();
    } catch (parseError) {
      data = {};
    }

    if (!response.ok) {
      showError(data.error || "The server returned an error. Please try again.");
      return;
    }

    showResult(data.prediction, data.confidence);
  } catch (networkError) {
    showError("Could not reach the server. Check your connection and try again.");
  } finally {
    setLoading(false);
  }
}

analyzeBtn.addEventListener("click", analyze);

clearBtn.addEventListener("click", () => {
  textInput.value = "";
  errorBox.hidden = true;
  resultCard.hidden = true;
  textInput.focus();
});

textInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey)) {
    analyze();
  }
});
