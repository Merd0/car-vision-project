"""API tests for the car vision FastAPI application.

The tests replace heavy model inference with lightweight doubles and verify:
- health endpoint shape
- analyze endpoint shape, including top-3 predictions
- generic fallback feature and valuation behavior for unknown classes
- input validation for unsupported uploads
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import Generator

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from car_vision_project.api import main as api_main
from car_vision_project.services.feature_service import CarIdentity, MockFeatureService
from car_vision_project.services.valuation_service import MockValuationService


class FakeVisionService:
    """Small inference double that avoids loading a real PyTorch checkpoint."""

    def __init__(self) -> None:
        self.checkpoint_path = Path("tests/fixtures/fake_model.pt")
        self.class_names = [
            "toyota_corolla_2020",
            "honda_civic_2019",
            "bmw_320i_2021",
        ]
        self.model = object()

    def load(self) -> None:
        """Simulate successful model startup."""

        self.model = object()

    def predict(
        self,
        image: Image.Image,
        top_k: int = 3,
    ) -> tuple[api_main.PredictionResponse, list[api_main.PredictionResponse]]:
        """Return deterministic primary and top-k predictions."""

        assert isinstance(image, Image.Image)
        assert top_k == 3

        top_predictions = [
            api_main.PredictionResponse(
                make="Toyota",
                model="Corolla",
                year=2020,
                class_label="toyota_corolla_2020",
                confidence=0.97,
            ),
            api_main.PredictionResponse(
                make="Honda",
                model="Civic",
                year=2019,
                class_label="honda_civic_2019",
                confidence=0.02,
            ),
            api_main.PredictionResponse(
                make="Bmw",
                model="320I",
                year=2021,
                class_label="bmw_320i_2021",
                confidence=0.01,
            ),
        ]
        return top_predictions[0], top_predictions


class FakeFeatureService:
    """Feature service double for isolated API tests."""

    def get_features(self, car: CarIdentity) -> dict[str, object]:
        assert car == CarIdentity(make="Toyota", model="Corolla", year=2020)
        return {
            "body_type": "Sedan",
            "engine": "1.8L Hybrid",
            "horsepower_hp": 121,
            "transmission": "CVT",
            "fuel_type": "Hybrid",
        }


class FakeValuationService:
    """Valuation service double for isolated API tests."""

    def get_valuation(self, car: CarIdentity) -> api_main.ValuationResponse:
        assert car == CarIdentity(make="Toyota", model="Corolla", year=2020)
        return api_main.ValuationResponse(
            currency="USD",
            current_market_value=22500.0,
            second_hand_market_value=17800.0,
            mileage_assumption_km=80000,
            source="test_double",
        )


@pytest.fixture()
def client(monkeypatch: pytest.MonkeyPatch) -> Generator[TestClient, None, None]:
    """Create a TestClient with all heavy or external dependencies mocked."""

    monkeypatch.setattr(api_main, "vision_service", FakeVisionService())
    monkeypatch.setattr(api_main, "feature_service", FakeFeatureService())
    monkeypatch.setattr(api_main, "valuation_service", FakeValuationService())

    with TestClient(api_main.app) as test_client:
        yield test_client


def test_health_returns_api_status(client: TestClient) -> None:
    """The health endpoint should report API and model availability."""

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["model_loaded"] is True
    assert payload["classes_loaded"] == 3
    assert payload["checkpoint_path"] == "tests/fixtures/fake_model.pt"
    assert payload["classes_path"].replace("\\", "/").endswith(
        "car_vision_project/artifacts/classes.txt"
    )


def test_analyze_car_returns_prediction_topk_features_and_valuation(
    client: TestClient,
) -> None:
    """The analyze endpoint should compose inference and service results."""

    response = client.post(
        "/analyze-car",
        files={"file": ("car.jpg", build_test_image_bytes(), "image/jpeg")},
    )

    assert response.status_code == 200
    payload = response.json()

    assert payload["prediction"] == {
        "make": "Toyota",
        "model": "Corolla",
        "year": 2020,
        "class_label": "toyota_corolla_2020",
        "confidence": 0.97,
    }
    assert len(payload["top_predictions"]) == 3
    assert payload["top_predictions"][0] == payload["prediction"]
    assert payload["top_predictions"][1]["class_label"] == "honda_civic_2019"
    assert payload["top_predictions"][2]["confidence"] == 0.01

    assert payload["features"]["engine"] == "1.8L Hybrid"
    assert payload["features"]["horsepower_hp"] == 121
    assert payload["valuation"] == {
        "currency": "USD",
        "current_market_value": 22500.0,
        "second_hand_market_value": 17800.0,
        "mileage_assumption_km": 80000,
        "source": "test_double",
    }


def test_analyze_car_rejects_non_image_upload(client: TestClient) -> None:
    """The analyze endpoint should reject unsupported upload content types."""

    response = client.post(
        "/analyze-car",
        files={"file": ("car.txt", b"not an image", "text/plain")},
    )

    assert response.status_code == 415
    assert response.json()["detail"] == (
        "Uploaded file must be one of: image/jpeg, image/png, image/webp."
    )


def test_feature_service_returns_generic_fallback_for_unknown_class() -> None:
    """Feature service should never raise for unknown classes."""

    service = MockFeatureService()
    features = service.get_features(CarIdentity(make="Skoda", model="Octavia", year=2022))

    assert features["body_type"] == "Passenger Vehicle"
    assert features["engine"] == "Not available in mock catalog"
    assert features["estimated_vehicle_age_years"] == 4
    assert features["data_source"] == "generic fallback for Skoda Octavia"


def test_valuation_service_returns_generic_fallback_for_unknown_class() -> None:
    """Valuation service should never raise for unknown classes."""

    service = MockValuationService()
    valuation = service.get_valuation(CarIdentity(make="Skoda", model="Octavia", year=2022))

    assert valuation.currency == "USD"
    assert valuation.current_market_value > 0
    assert valuation.second_hand_market_value > 0
    assert valuation.mileage_assumption_km == 60000
    assert valuation.source == "generic_fallback_for_skoda_octavia"


def build_test_image_bytes() -> bytes:
    """Return a tiny valid JPEG image for upload tests."""

    image = Image.new("RGB", (32, 32), color=(240, 240, 240))
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG")
    return buffer.getvalue()
