import os
import json
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score


def load_predictions(file_path: str, model_name: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    df["model"] = model_name
    return df


def evaluate_model(df: pd.DataFrame) -> dict:
    y_true = df["true_label"]
    y_pred = df["predicted_label"]
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average="weighted")
    return {
        "model": df["model"].iloc[0],
        "accuracy": float(accuracy),
        "f1_score": float(f1),
    }


def compare_models(prediction_files: dict) -> pd.DataFrame:
    results = []
    for model_name, file_path in prediction_files.items():
        df = load_predictions(file_path, model_name)
        metrics = evaluate_model(df)
        results.append(metrics)
    return pd.DataFrame(results)


def plot_comparison(results: pd.DataFrame, save_path: str = None):
    """
    Note: do NOT mutate `results` in-place. Work on a copy.
    """
    df = results.set_index("model")  # works on a new object
    ax = df.plot(kind="bar", figsize=(8, 6))
    ax.set_title("Model Comparison on IMDB Dataset")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    plt.xticks(rotation=0)
    plt.legend(loc="lower right")

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
    plt.close()  # close to avoid interactive popups when running in CI


def main():
    prediction_files = {
        "Logistic Regression": "Data/predictions/logistic_preds.csv",
        "BiLSTM": "Data/predictions/bilstm_preds.csv",
        "DeBERTa": "Data/predictions/deberta_preds.csv",
    }

    # compute metrics table
    results = compare_models(prediction_files)

    print("\n=== Model Comparison Results ===\n")
    print(results.to_string(index=False))

    # save plot (does NOT modify `results`)
    plot_comparison(results, save_path="Data/predictions/model_comparison.png")

    # Save CSV summary
    os.makedirs("Data/predictions", exist_ok=True)
    results.to_csv("Data/predictions/comparison_results.csv", index=False)

    # Ensure metrics dir exists
    os.makedirs("metrics", exist_ok=True)

    # map display model names to the filenames DVC expects
    filename_map = {
        "Logistic Regression": "logistic_model.json",
        "BiLSTM": "bilstm_model.json",
        "DeBERTa": "deberta_model.json",
    }

    # write one JSON per model (for DVC)
    for _, row in results.iterrows():
        model_display = row["model"]
        fname = filename_map.get(model_display)
        if not fname:
            # fallback name
            fname = model_display.lower().replace(" ", "_") + "_model.json"

        metrics_path = os.path.join("metrics", fname)
        payload = {
            "accuracy": float(row["accuracy"]),
            "f1_score": float(row["f1_score"]),
        }
        with open(metrics_path, "w") as f:
            json.dump(payload, f, indent=4)

    print("\n✅ Metrics JSON files saved in 'metrics/'")
    print("✅ Comparison CSV saved at Data/predictions/comparison_results.csv")
    print("✅ Plot saved at Data/predictions/model_comparison.png")


if __name__ == "__main__":
    main()
