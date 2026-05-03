import json
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


TRAIN_PATH = Path("data/processed/train.csv")
TEST_PATH = Path("data/processed/test.csv")
REPORTS_DIR = Path("reports")
COMPARISON_PATH = REPORTS_DIR / "model_comparison.csv"


def evaluate_model(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    mae = mean_absolute_error(y_test, predictions)
    rmse = mean_squared_error(y_test, predictions) ** 0.5
    r2 = r2_score(y_test, predictions)

    return mae, rmse, r2


def main():
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            "Training data not found. Run python src/preprocess.py first."
        )

    if not TEST_PATH.exists():
        raise FileNotFoundError(
            "Testing data not found. Run python src/preprocess.py first."
        )

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = pd.read_csv(TRAIN_PATH)
    test_df = pd.read_csv(TEST_PATH)

    X_train = train_df.drop("MedHouseVal", axis=1)
    y_train = train_df["MedHouseVal"]

    X_test = test_df.drop("MedHouseVal", axis=1)
    y_test = test_df["MedHouseVal"]

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1,
        ),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    }

    results = []

    mlflow.set_experiment("house-price-model-comparison")

    for model_name, model in models.items():
        with mlflow.start_run(run_name=model_name):
            mae, rmse, r2 = evaluate_model(
                model,
                X_train,
                y_train,
                X_test,
                y_test,
            )

            mlflow.log_param("model_name", model_name)
            mlflow.log_metric("mae", mae)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("r2", r2)
            mlflow.sklearn.log_model(model, name=f"{model_name.lower().replace(' ', '_')}_model")

            results.append(
                {
                    "model": model_name,
                    "mae": round(mae, 4),
                    "rmse": round(rmse, 4),
                    "r2": round(r2, 4),
                }
            )

    results_df = pd.DataFrame(results).sort_values(by="rmse")
    results_df.to_csv(COMPARISON_PATH, index=False)

    print("Model comparison complete.")
    print(results_df)
    print(f"Saved comparison report to {COMPARISON_PATH}")


if __name__ == "__main__":
    main()