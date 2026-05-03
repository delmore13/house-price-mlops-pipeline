from fastapi.testclient import TestClient

from app.main import app


client = TestClient(app)


def test_home_endpoint():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "House Price MLOps API is running"


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict_endpoint():
    sample_payload = {
        "MedInc": 8.3252,
        "HouseAge": 41.0,
        "AveRooms": 6.984127,
        "AveBedrms": 1.02381,
        "Population": 322.0,
        "AveOccup": 2.555556,
        "Latitude": 37.88,
        "Longitude": -122.23,
    }

    response = client.post("/predict", json=sample_payload)

    assert response.status_code == 200

    response_body = response.json()

    assert "input_features" in response_body
    assert "prediction" in response_body
    assert "predicted_house_value" in response_body["prediction"]
    assert isinstance(response_body["prediction"]["predicted_house_value"], float)