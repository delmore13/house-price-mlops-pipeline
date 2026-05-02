from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TRAIN_DATA_PATH = Path("data/processed/train.csv")
TEST_DATA_PATH = Path("data/processed/test.csv")
MODEL_DIR = Path("models")
MODEL_PATH = MODEL_DIR / "house_price_model.joblib"

TARGET_COLUMN = "MedHouseVal"


def train_model(
    train_data_path: Path = TRAIN_DATA_PATH,
    test_data_path: Path = TEST_DATA_PATH,
    model_path: Path = MODEL_PATH,
) -> RandomForestRegressor:
    """
    Train a Random Forest model, evaluate it, save the model artifact,
    and log parameters, metrics, and model artifacts with MLflow.
    """
    if not train_data_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {train_data_path}. "
            "Run `python src/preprocess.py` first."
        )

    if not test_data_path.exists():
        raise FileNotFoundError(
            f"Test data not found at {test_data_path}. "
            "Run `python src/preprocess.py` first."
        )

    train_df = pd.read_csv(train_data_path)
    test_df = pd.read_csv(test_data_path)

    if TARGET_COLUMN not in train_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in training data.")

    if TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in test data.")

    X_train = train_df.drop(columns=[TARGET_COLUMN])
    y_train = train_df[TARGET_COLUMN]

    X_test = test_df.drop(columns=[TARGET_COLUMN])
    y_test = test_df[TARGET_COLUMN]

    params = {
        "n_estimators": 100,
        "random_state": 42,
        "n_jobs": -1,
    }

    mlflow.set_experiment("house-price-mlops-pipeline")

    with mlflow.start_run():
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)

        predictions = model.predict(X_test)

        mae = mean_absolute_error(y_test, predictions)
        mse = mean_squared_error(y_test, predictions)
        rmse = mse ** 0.5
        r2 = r2_score(y_test, predictions)

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)

        mlflow.log_params(params)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2_score", r2)

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="random_forest_model",
        )

        print("Model trained successfully with MLflow tracking.")
        print(f"Saved model to {model_path}")
        print(
            {
                "mae": round(mae, 4),
                "rmse": round(rmse, 4),
                "r2_score": round(r2, 4),
            }
        )

        return model


if __name__ == "__main__":
    train_model()