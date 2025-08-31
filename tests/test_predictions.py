import os
import pytest
import pandas as pd

# Path to predictions folder
PREDICTIONS_DIR = "Data/predictions"

# Expected prediction files
PRED_FILES = {
    "logistic": "logistic_preds.csv",
    "bilstm": "bilstm_preds.csv",
    "deberta": "deberta_preds.csv",
}


@pytest.mark.parametrize("model_name, filename", PRED_FILES.items())
def test_prediction_file_exists(model_name, filename):
    """Check that the prediction file exists"""
    filepath = os.path.join(PREDICTIONS_DIR, filename)
    assert os.path.exists(filepath), f"{model_name} predictions file is missing!"


@pytest.mark.parametrize("model_name, filename", PRED_FILES.items())
def test_prediction_file_not_empty(model_name, filename):
    """Check that the prediction file is not empty"""
    filepath = os.path.join(PREDICTIONS_DIR, filename)
    df = pd.read_csv(filepath)
    assert not df.empty, f"{model_name} predictions file is empty!"


@pytest.mark.parametrize("model_name, filename", PRED_FILES.items())
def test_prediction_columns_exist(model_name, filename):
    """Check that required columns y_true and y_pred exist"""
    filepath = os.path.join(PREDICTIONS_DIR, filename)
    df = pd.read_csv(filepath)
    required_columns = {"true_label", "predicted_label"}
    assert required_columns.issubset(df.columns), (
        f"{model_name} file must contain columns {required_columns}, "
        f"but found {set(df.columns)}"
    )


@pytest.mark.parametrize("model_name, filename", PRED_FILES.items())
def test_prediction_lengths_match(model_name, filename):
    """Check that y_true and y_pred have the same length"""
    filepath = os.path.join(PREDICTIONS_DIR, filename)
    df = pd.read_csv(filepath)
    assert len(df["true_label"]) == len(df["predicted_label"]), (
        f"{model_name} file has mismatched y_true and y_pred lengths!"
    )
