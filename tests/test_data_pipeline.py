from pathlib import Path

import pandas as pd

from src.ingest_data import ingest_data
from src.preprocess import preprocess_data
from src.train import train_model
from src.evaluate import evaluate_model


def test_ingest_data_creates_raw_file():
    output_path = Path("data/raw/test_housing.csv")

    df = ingest_data(output_path=output_path)

    assert output_path.exists()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "MedHouseVal" in df.columns

    output_path.unlink()


def test_preprocess_data_creates_train_and_test_files():
    train_path = Path("data/processed/test_train.csv")
    test_path = Path("data/processed/test_test.csv")

    train_df, test_df = preprocess_data(
        train_data_path=train_path,
        test_data_path=test_path,
    )

    assert train_path.exists()
    assert test_path.exists()
    assert not train_df.empty
    assert not test_df.empty
    assert "MedHouseVal" in train_df.columns
    assert "MedHouseVal" in test_df.columns

    train_path.unlink()
    test_path.unlink()


def test_train_model_creates_model_file():
    model_path = Path("models/test_house_price_model.joblib")

    model = train_model(model_path=model_path)

    assert model_path.exists()
    assert hasattr(model, "predict")

    model_path.unlink()


def test_evaluate_model_creates_metrics_file():
    metrics_path = Path("reports/test_metrics.json")

    metrics = evaluate_model(metrics_path=metrics_path)

    assert metrics_path.exists()
    assert "mae" in metrics
    assert "rmse" in metrics
    assert "r2_score" in metrics
    assert metrics["r2_score"] > 0

    metrics_path.unlink()