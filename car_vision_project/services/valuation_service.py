"""Mock valuation service for current and second-hand vehicle prices."""

from __future__ import annotations

from dataclasses import dataclass

from car_vision_project.services.feature_service import CarIdentity


@dataclass(frozen=True)
class VehicleValuation:
    """Market valuation details for a vehicle."""

    currency: str
    current_market_value: float
    second_hand_market_value: float
    mileage_assumption_km: int
    source: str


class ValuationNotFoundError(LookupError):
    """Raised when valuation data is unavailable."""


class MockValuationService:
    """Simulates fetching price data from a market valuation provider."""

    def __init__(self) -> None:
        self._valuations: dict[tuple[str, str, int], VehicleValuation] = {
            ("toyota", "corolla", 2020): VehicleValuation(
                currency="USD",
                current_market_value=22500.0,
                second_hand_market_value=17800.0,
                mileage_assumption_km=80000,
                source="mock_market_index",
            ),
            ("honda", "civic", 2019): VehicleValuation(
                currency="USD",
                current_market_value=21500.0,
                second_hand_market_value=16500.0,
                mileage_assumption_km=90000,
                source="mock_market_index",
            ),
            ("bmw", "320i", 2021): VehicleValuation(
                currency="USD",
                current_market_value=39800.0,
                second_hand_market_value=31500.0,
                mileage_assumption_km=70000,
                source="mock_market_index",
            ),
            ("tesla", "model 3", 2022): VehicleValuation(
                currency="USD",
                current_market_value=38900.0,
                second_hand_market_value=31800.0,
                mileage_assumption_km=60000,
                source="mock_market_index",
            ),
        }

    def get_valuation(self, car: CarIdentity) -> VehicleValuation:
        """Return valuation data for a car identity."""

        valuation = self._valuations.get(car.key)
        if valuation is not None:
            return valuation
        return self._build_fallback_valuation(car)

    @staticmethod
    def _build_fallback_valuation(car: CarIdentity) -> VehicleValuation:
        """Return a generic valuation payload for classes without mock data."""

        age = max(0, 2026 - car.year)
        current_market_value = max(9000.0, 36000.0 - age * 1800.0)
        second_hand_market_value = max(7000.0, current_market_value * 0.82)
        mileage_assumption_km = 15000 * max(1, age)

        return VehicleValuation(
            currency="USD",
            current_market_value=round(current_market_value, 2),
            second_hand_market_value=round(second_hand_market_value, 2),
            mileage_assumption_km=mileage_assumption_km,
            source=f"generic_fallback_for_{car.make.lower()}_{car.model.lower().replace(' ', '_')}",
        )
