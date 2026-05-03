from fastapi import FastAPI
from pydantic import BaseModel, Field
from src.predict import predict_house_price


app = FastAPI(
    title="House Price MLOps API",
    description="A production-style machine learning API for California housing price predictions.",
    version="1.0.0",
)


class HouseFeatures(BaseModel):
    MedInc: float = Field(..., example=8.3252)
    HouseAge: float = Field(..., example=41.0)
    AveRooms: float = Field(..., example=6.984127)
    AveBedrms: float = Field(..., example=1.023810)
    Population: float = Field(..., example=322.0)
    AveOccup: float = Field(..., example=2.555556)
    Latitude: float = Field(..., example=37.88)
    Longitude: float = Field(..., example=-122.23)


@app.get("/")
def home():
    return {
        "message": "House Price MLOps API is running",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": True,
    }


@app.post("/predict")
def predict(features: HouseFeatures):
    prediction = predict_house_price(features.dict())

    return {
        "input_features": features.dict(),
        "prediction": prediction,
    }