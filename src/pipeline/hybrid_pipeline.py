import joblib
import numpy as np
import yaml
from pathlib import Path

from src.nlp.preprocessing import clean_text
from src.llm.llm_analyzer import analyze_with_llm
from src.pipeline.domain_guard import is_banking_query


BASE_DIR = Path(__file__).resolve().parents[2]

with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as file:
    config = yaml.safe_load(file)

CONFIDENCE_THRESHOLD = float(
    config["pipeline"]["confidence_threshold"]
)

svm_model = joblib.load(BASE_DIR / "models" / "svm_model.pkl")
tfidf_vectorizer = joblib.load(BASE_DIR / "models" / "tfidf_vectorizer.pkl")

def predict_intent(text):
    clean = clean_text(text)
    vector = tfidf_vectorizer.transform([clean])

    prediction = svm_model.predict(vector)[0]
    scores = svm_model.decision_function(vector)[0]

    sorted_scores = np.sort(scores)[::-1]
    margin = sorted_scores[0] - sorted_scores[1]

    return prediction, margin


def analyze_customer_query(customer_query):
    if not is_banking_query(customer_query):
        return {
            "route": "OUT_OF_SCOPE",
            "intent": None,
            "margin": None,
            "llm_analysis": None
        }

    intent, margin = predict_intent(customer_query)

    if margin >= CONFIDENCE_THRESHOLD:
        return {
            "route": "ML",
            "intent": intent,
            "margin": float(margin),
            "llm_analysis": None
        }

    llm_result = analyze_with_llm(
    customer_query,
    list(svm_model.classes_)
)

    return {
        "route": "LLM",
        "intent": llm_result.intent,
        "margin": float(margin),
        "llm_analysis": llm_result
    }