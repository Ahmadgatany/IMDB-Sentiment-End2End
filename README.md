# 🎬 IMDB-Sentiment-End2End

A professional **Sentiment Analysis** project on **IMDB Reviews** using multiple models: **Logistic Regression, BiLSTM, and DeBERTa**.

The project follows **MLOps practices** including:

* **DVC** for data and model versioning
* **MLflow** for experiment tracking
* **CI/CD** with GitHub Actions
* **FastAPI + Docker** for deployment

---

## 📂 Project Structure

```
NLP-Sentiment-IMDB/
├── .dvc/                  ← Internal DVC files
├── .github/workflows/
│   └── mlflow.yml         ← CI/CD workflow for MLflow & deployment
├── config/
│   └── config.yaml        ← Paths, hyperparameters, and settings
├── data/
│   ├── raw/               ← Original IMDB dataset
│   ├── processed/         ← Cleaned & split dataset
│   ├── predictions/       ← Model predictions
├── deployment/
│   ├── app.py             ← FastAPI application
│   ├── Dockerfile
│   ├── index.html
│   ├── requirements.txt
├── mlruns/                ← MLflow tracking logs
├── models/
│   ├── logistic_model.pkl
│   ├── bilstm_model.h5
│   ├── deberta_model/
├── notebooks/
│   ├── classic_models.ipynb
│   ├── transformer_deberta.ipynb
│   └── comparison_analysis.ipynb
├── src/
│   ├── data_preprocessing.py
│   ├── train_logistic.py
│   ├── train_bilstm.py
│   ├── train_deberta.py
│   └── compare_models.py
├── tests/
│   └── test_predictions.py
├── dvc.yaml               ← DVC pipeline stages
├── .gitignore
├── .dvcignore
└── README.md
```

---

## 🚀 Features

* **Multiple Models**: Logistic Regression, BiLSTM, DeBERTa
* **Organized Pipeline** with DVC
* **Experiment Tracking** via MLflow
* **Model Comparison** (accuracy, F1, etc.)
* **Deployment-ready API** with FastAPI
* **Containerization** with Docker
* **CI/CD** using GitHub Actions

---

## ⚙️ Usage

### 1. Setup Environment

```bash
pip install -r deployment/requirements.txt
```

### 2. Run the Pipeline with DVC

```bash
dvc repro
```

### 3. Launch MLflow UI

```bash
mlflow ui
```

### 4. Run the API

```bash
uvicorn deployment.app:app --reload
```

---

## 🧪 Testing

Run unit tests:

```bash
pytest tests/
```

---

## 📊 Results

* **Logistic Regression**: baseline model
* **BiLSTM**: deep learning sequence model
* **DeBERTa**: state-of-the-art transformer

📌 Final metrics will be updated after running all experiments and logging them in **MLflow**.

---

## 🤝 Contribution

* Fork & submit Pull Requests are welcome
* Follow **PEP8** and keep code clean
* Use **branching strategy** (`feature/`, `fix/`)

---

## 📝 License

MIT License © 2025
