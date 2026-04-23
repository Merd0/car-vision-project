"""Centralized application configuration.

All runtime settings should be read through ``get_settings()`` so local runs,
tests, and Docker deployments use the same environment variable contract.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


BASE_DIR = Path(__file__).resolve().parent


class AppSettings(BaseSettings):
    """Environment-backed configuration for the car vision service."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    app_env: str = "production"
    log_level: str = "info"

    api_host: str = "0.0.0.0"
    api_port: int = Field(default=8000, ge=1, le=65535)
    uvicorn_workers: int = Field(default=1, ge=1)
    cors_allowed_origins: str = ""

    model_checkpoint_path: Path | None = BASE_DIR / "artifacts" / "best_car_model.pth"
    model_classes_path: Path = BASE_DIR / "artifacts" / "classes.txt"
    model_num_classes: int = Field(default=105, gt=1)
    model_image_size: int = Field(default=224, gt=0)
    model_device: Literal["auto", "cpu", "cuda"] = "auto"
    model_dropout: float = Field(default=0.3, ge=0.0, lt=1.0)

    templates_dir: Path = BASE_DIR / "templates"
    static_dir: Path = BASE_DIR / "static"

    mock_db_url: str = "mock://local-car-market"
    feature_service_timeout_seconds: float = Field(default=2.0, gt=0)
    valuation_service_timeout_seconds: float = Field(default=2.0, gt=0)

    max_upload_size_mb: int = Field(default=10, gt=0)
    allowed_image_content_types: str = "image/jpeg,image/png,image/webp"

    @field_validator("model_checkpoint_path", mode="before")
    @classmethod
    def normalize_model_checkpoint_path(cls, value: object) -> object:
        """Treat an empty MODEL_CHECKPOINT_PATH as an intentional demo mode."""

        if isinstance(value, str) and not value.strip():
            return None
        return value

    @field_validator("allowed_image_content_types")
    @classmethod
    def validate_content_types(cls, value: str) -> str:
        """Ensure upload content types are configured as a non-empty CSV string."""

        values = [item.strip() for item in value.split(",") if item.strip()]
        if not values:
            raise ValueError("allowed_image_content_types must contain at least one value.")
        invalid_values = [item for item in values if "/" not in item]
        if invalid_values:
            raise ValueError(
                "allowed_image_content_types entries must look like MIME types: "
                f"{invalid_values}."
            )
        return ",".join(values)

    @property
    def max_upload_size_bytes(self) -> int:
        """Return the maximum accepted upload size in bytes."""

        return self.max_upload_size_mb * 1024 * 1024

    @property
    def allowed_image_content_type_values(self) -> tuple[str, ...]:
        """Return configured upload MIME types as a tuple."""

        return tuple(
            item.strip()
            for item in self.allowed_image_content_types.split(",")
            if item.strip()
        )

    @property
    def cors_allowed_origin_values(self) -> tuple[str, ...]:
        """Return configured CORS origins as a tuple."""

        return tuple(
            item.strip()
            for item in self.cors_allowed_origins.split(",")
            if item.strip()
        )

    def resolve_model_device(self, cuda_available: bool) -> str:
        """Resolve ``auto`` model device into ``cuda`` or ``cpu``."""

        if self.model_device == "auto":
            return "cuda" if cuda_available else "cpu"
        return self.model_device


@lru_cache(maxsize=1)
def get_settings() -> AppSettings:
    """Return cached application settings."""

    return AppSettings()
