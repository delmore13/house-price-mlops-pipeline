from pathlib import Path
import json

import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TEST_DATA_PATH = Path("data/processed/test.csv")
MODEL_PATH = Path("models/house_price_model.joblib")
REPORTS_DIR = Path("reports")
METRICS_PATH = REPORTS_DIR / "metrics.json"

TARGET_COLUMN = "MedHouseVal"


def evaluate_model(
    test_data_path: Path = TEST_DATA_PATH,
    model_path: Path = MODEL_PATH,
    metrics_path: Path = METRICS_PATH,
) -> dict:
    """
    Load the trained model and test dataset, evaluate performance,
    and save metrics to a JSON report.
    """
    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_data_path}. "
            "Run `python src/preprocess.py` first."
        )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model file not found at {model_path}. "
            "Run `python src/train.py` first."
        )

    test_df = pd.read_csv(test_data_path)

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found.")

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    model = joblib.load(model_path)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = mse ** 0.5
    r2 = r2_score(y_test, predictions)

    metrics = {
        "model": "RandomForestRegressor",
        "mae": round(mae, 4),
        "rmse": round(rmse, 4),
        "r2_score": round(r2, 4),
    }

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    with open(metrics_path, "w") as file:
        json.dump(metrics, file, indent=4)

    return metrics


if __name__ == "__main__":
    metrics = evaluate_model()
    print("Model evaluation complete.")
    print(f"Saved metrics to {METRICS_PATH}")
    print(metrics)