from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import joblib
import tensorflow as tf
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

# -----------------------------
# Initialize FastAPI
# -----------------------------
app = FastAPI(
    title="IMDB Sentiment Analysis API",
    description="Serve Logistic Regression, BiLSTM, and DeBERTa models",
    version="1.0"
)

# -----------------------------
# Request Schema
# -----------------------------
class ReviewRequest(BaseModel):
    text: str
    model_name: str  # "logistic", "bilstm", "deberta"

# -----------------------------
# Load Models
# -----------------------------
# Logistic Regression + TF-IDF
logistic_model = joblib.load("models/logistic_model.pkl")
tfidf_vectorizer = joblib.load("models/tfidf_vectorizer.pkl")

# BiLSTM + Tokenizer
bilstm_model = tf.keras.models.load_model("models/bilstm_model.h5")
tokenizer_bilstm = joblib.load("models/tokenizer_bilstm.pkl")

# DeBERTa
deberta_tokenizer = AutoTokenizer.from_pretrained("models/deberta")
deberta_model = AutoModelForSequenceClassification.from_pretrained("models/deberta")

# -----------------------------
# Helper Functions
# -----------------------------
def predict_logistic(text: str):
    X = tfidf_vectorizer.transform([text])
    pred = logistic_model.predict(X)[0]
    return "positive" if pred == 1 else "negative"

def predict_bilstm(text: str, max_len=200):
    seq = tokenizer_bilstm.texts_to_sequences([text])
    padded = tf.keras.preprocessing.sequence.pad_sequences(seq, maxlen=max_len)
    pred = bilstm_model.predict(padded)
    return "positive" if pred[0][0] > 0.5 else "negative"

def predict_deberta(text: str):
    inputs = deberta_tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=256)
    with torch.no_grad():
        outputs = deberta_model(**inputs)
        probs = torch.softmax(outputs.logits, dim=-1)
        pred = torch.argmax(probs, dim=1).item()
    return "positive" if pred == 1 else "negative"

# -----------------------------
# API Endpoint
# -----------------------------
@app.post("/predict")
def predict(request: ReviewRequest):
    model_name = request.model_name.lower()
    text = request.text

    if model_name == "logistic":
        result = predict_logistic(text)
    elif model_name == "bilstm":
        result = predict_bilstm(text)
    elif model_name == "deberta":
        result = predict_deberta(text)
    else:
        raise HTTPException(status_code=400, detail="Invalid model_name. Choose from: logistic, bilstm, deberta")

    return {"model": model_name, "text": text, "prediction": result}


# -----------------------------
# Serve index.html
# -----------------------------
@app.get("/")
def read_index():
    return FileResponse(os.path.join(os.path.dirname(__file__), "index.html"))
