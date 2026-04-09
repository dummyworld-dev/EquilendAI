import os

import numpy as np
import pandas as pd

from src.models.train_random_forest import load_prepared_data as load_rf_data, train_random_forest
from src.models.train_xgboost import load_prepared_data as load_xgb_data, train_xgboost_classifier


def test_load_prepared_data_shapes():
    csv_path = "data/equilend_mock_data.csv"
    if not os.path.exists(csv_path):
        # Skip if synthetic data has not been generated yet.
        return

    df_rf = load_rf_data(csv_path)
    df_xgb = load_xgb_data(csv_path)

    # Both pipelines should produce dataframes with the target column present.
    assert "default_status" in df_rf.columns
    assert "default_status" in df_xgb.columns
    assert len(df_rf) > 0
    assert len(df_xgb) > 0


def test_random_forest_training_creates_model_file(tmp_path):
    csv_path = "data/equilend_mock_data.csv"
    if not os.path.exists(csv_path):
        return

    model_path = tmp_path / "rf_model.joblib"
    train_random_forest(csv_path=csv_path, model_output_path=str(model_path), test_size=0.3)

    assert model_path.exists()


def test_xgboost_training_returns_metrics(tmp_path):
    csv_path = "data/equilend_mock_data.csv"
    if not os.path.exists(csv_path):
        return

    model_path = tmp_path / "xgb_model.joblib"
    metrics = train_xgboost_classifier(
        csv_path=csv_path,
        model_output_path=str(model_path),
        test_size=0.3,
        use_smote=False,
    )

    assert model_path.exists()
    # Check key metrics are present and within valid ranges.
    assert "accuracy" in metrics and 0.0 <= metrics["accuracy"] <= 1.0
    assert "roc_auc" in metrics or np.isnan(metrics["roc_auc"])

