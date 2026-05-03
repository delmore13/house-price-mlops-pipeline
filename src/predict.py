import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path("models/house_price_model.joblib")


def predict_house_price(input_data: dict):
    """
    Loads the trained model and returns a house price prediction.
    """

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found. Run python src/train.py first."
        )

    model = joblib.load(MODEL_PATH)

    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]

    return {
        "predicted_house_value": round(float(prediction), 2)
    }


if __name__ == "__main__":
    sample_house = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.023810,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23
    }

    result = predict_house_price(sample_house)
    print(result)