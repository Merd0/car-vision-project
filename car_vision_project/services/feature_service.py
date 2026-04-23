"""Mock feature service for vehicle technical specifications."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CarIdentity:
    """Structured car identity produced by the vision model."""

    make: str
    model: str
    year: int

    @property
    def key(self) -> tuple[str, str, int]:
        """Return a normalized lookup key."""

        return (self.make.lower(), self.model.lower(), self.year)


class FeatureNotFoundError(LookupError):
    """Raised when vehicle features are unavailable."""


class MockFeatureService:
    """Simulates fetching technical features from a vehicle database."""

    def __init__(self) -> None:
        self._features: dict[tuple[str, str, int], dict[str, Any]] = {
            ("toyota", "corolla", 2020): {
                "body_type": "Sedan",
                "engine": "1.8L Hybrid",
                "horsepower_hp": 121,
                "transmission": "CVT",
                "fuel_type": "Hybrid",
                "drivetrain": "FWD",
                "average_consumption_l_100km": 4.5,
            },
            ("honda", "civic", 2019): {
                "body_type": "Sedan",
                "engine": "1.5L Turbo",
                "horsepower_hp": 174,
                "transmission": "CVT",
                "fuel_type": "Gasoline",
                "drivetrain": "FWD",
                "average_consumption_l_100km": 6.7,
            },
            ("bmw", "320i", 2021): {
                "body_type": "Sedan",
                "engine": "2.0L Turbo",
                "horsepower_hp": 184,
                "transmission": "8-Speed Automatic",
                "fuel_type": "Gasoline",
                "drivetrain": "RWD",
                "average_consumption_l_100km": 6.4,
            },
            ("tesla", "model 3", 2022): {
                "body_type": "Sedan",
                "engine": "Dual Motor Electric",
                "horsepower_hp": 425,
                "transmission": "Single-Speed Automatic",
                "fuel_type": "Electric",
                "drivetrain": "AWD",
                "estimated_range_km": 576,
            },
        }

    def get_features(self, car: CarIdentity) -> dict[str, Any]:
        """Return technical features for a car identity."""

        features = self._features.get(car.key)
        if features is not None:
            return dict(features)
        return self._build_fallback_features(car)

    @staticmethod
    def _build_fallback_features(car: CarIdentity) -> dict[str, Any]:
        """Return a generic feature payload for classes without mock data."""

        current_year = 2026
        vehicle_age = max(0, current_year - car.year)

        return {
            "body_type": "Passenger Vehicle",
            "engine": "Not available in mock catalog",
            "horsepower_hp": "Unknown",
            "transmission": "Unknown",
            "fuel_type": "Unknown",
            "drivetrain": "Unknown",
            "estimated_vehicle_age_years": vehicle_age,
            "data_source": f"generic fallback for {car.make} {car.model}",
        }


def parse_class_label(class_label: str) -> CarIdentity:
    """Parse a classifier label like ``toyota_corolla_2020`` into fields."""

    parts = class_label.strip().replace("-", "_").split("_")
    if len(parts) < 3:
        raise ValueError(
            "Class label must follow '<make>_<model>_<year>', "
            f"received '{class_label}'."
        )

    year_text = parts[-1]
    make = parts[0]
    model = "_".join(parts[1:-1])

    try:
        year = int(year_text)
    except ValueError as exc:
        raise ValueError(f"Class label year is invalid: '{year_text}'.") from exc

    return CarIdentity(make=make.title(), model=model.replace("_", " ").title(), year=year)
