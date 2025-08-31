# 🎬 IMDB Sentiment End-to-End

This project implements an **End-to-End Machine Learning Pipeline** for sentiment analysis on the IMDB movie reviews dataset.  
It covers the full lifecycle of an ML project: from **data preprocessing**, **model training**, **evaluation**, to **deployment** using FastAPI & Docker.  

---

## 🚀 Features
- Preprocessing pipeline with DVC for reproducibility.  
- Multiple models implemented and compared:
  - Logistic Regression
  - BiLSTM
  - DeBERTa (transformer-based)
- Experiment tracking with **MLflow**.  
- Deployment-ready FastAPI app with Docker.  
- Organized project structure (config-driven).  

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

## ⚙️ Installation & Usage

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
Then visit 👉 http://127.0.0.1:8000

### 5. Clone the repository
```bash
git clone https://github.com/Ahmadgatany/IMDB-Sentiment-End2End.git
cd IMDB-Sentiment-End2End
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

| Model               | Accuracy | Notes                    |
| ------------------- | -------- | ------------------------ |
| Logistic Regression | \~86%    | Baseline                 |
| BiLSTM              | \~89%    | Sequence model           |
| DeBERTa             | \~95%    | Transformer (fine-tuned) |


📌 Final metrics will be updated after running all experiments and logging them in **MLflow**.

---

## 🤝 Contribution

* Fork & submit Pull Requests are welcome
* Follow **PEP8** and keep code clean
* Use **branching strategy** (`feature/`, `fix/`)

---

## 📝 License

MIT License © 2025
