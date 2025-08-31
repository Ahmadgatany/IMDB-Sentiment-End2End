import pandas as pd
import numpy as np
import joblib
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import mlflow
import mlflow.sklearn

# 1. Load Config
with open("config/config.yaml") as f:
    config = yaml.safe_load(f)
config["tfidf_params"]["ngram_range"] = tuple(config["tfidf_params"]["ngram_range"])

# 2. Load Data
train_df = pd.read_csv("data/processed/train.csv")
val_df = pd.read_csv("data/processed/val.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train = train_df["clean_review"]
y_train = train_df["sentiment"]

X_val = val_df["clean_review"]
y_val = val_df["sentiment"]

X_test = test_df["clean_review"]
y_test = test_df["sentiment"]

# 3. TF-IDF Vectorizer (fit on train only)
tfidf = TfidfVectorizer(**config["tfidf_params"])
X_train_tfidf = tfidf.fit_transform(X_train)
X_val_tfidf = tfidf.transform(X_val)
X_test_tfidf = tfidf.transform(X_test)

# 4. Train Logistic Regression
model = LogisticRegression(**config["logistic_regression"])
model.fit(X_train_tfidf, y_train)

# 5. Evaluate on Validation Set
y_val_pred = model.predict(X_val_tfidf)
acc = accuracy_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)

# 6. MLflow Logging
with mlflow.start_run(run_name="Logistic Regression"):
    mlflow.set_tag("model_type", "TF-IDF + LogisticRegression")
    mlflow.log_params(config["logistic_regression"])
    mlflow.log_metric("val_accuracy", acc)
    mlflow.log_metric("val_f1_score", f1)
    mlflow.log_metric("val_precision", precision)
    mlflow.log_metric("val_recall", recall)

    mlflow.sklearn.log_model(model, artifact_path="logistic_model")
    mlflow.log_artifact("config/config.yaml")

# 7. Save model and vectorizer locally
joblib.dump(model, "models/logistic_model.pkl")
joblib.dump(tfidf, "models/tfidf_vectorizer.pkl")

# 8. Generate predictions on test set and save
y_test_pred = model.predict(X_test_tfidf)
y_test_probs = model.predict_proba(X_test_tfidf)

results_df = pd.DataFrame({
    "review": test_df["review"],
    "true_label": y_test,
    "predicted_label": y_test_pred,
    "prob_negative": y_test_probs[:, 0],
    "prob_positive": y_test_probs[:, 1]
})

results_df.to_csv("Data/predictions/logistic_preds.csv", index=False)
mlflow.log_artifact("Data/predictions/logistic_preds.csv")

print("✅ Logistic Regression training complete. Model, vectorizer, and predictions saved.")

