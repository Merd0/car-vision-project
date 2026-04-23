"""FastAPI application exposing car image analysis endpoints."""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any

import torch
from fastapi import FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, Field
from torch import nn
from torchvision import models

from car_vision_project.config import AppSettings, get_settings
from car_vision_project.services.feature_service import (
    CarIdentity,
    MockFeatureService,
    parse_class_label,
)
from car_vision_project.services.valuation_service import (
    MockValuationService,
)
from car_vision_project.utils.image_transforms import preprocess_pil_image


LOGGER = logging.getLogger(__name__)
SETTINGS = get_settings()
templates = Jinja2Templates(directory=str(SETTINGS.templates_dir))


class PredictionResponse(BaseModel):
    """Vision model prediction details."""

    make: str
    model: str
    year: int
    class_label: str
    confidence: float = Field(ge=0.0, le=1.0)


class AnalyzeCarResponse(BaseModel):
    """Response returned by the car analysis endpoint."""

    prediction: PredictionResponse
    top_predictions: list[PredictionResponse]
    features: dict[str, Any]
    valuation: ValuationResponse


class ValuationResponse(BaseModel):
    """Serializable valuation payload."""

    currency: str
    current_market_value: float
    second_hand_market_value: float
    mileage_assumption_km: int
    source: str


class VisionModelUnavailableError(RuntimeError):
    """Raised when inference is requested before a model is available."""


class VisionInferenceService:
    """Loads a trained classifier and performs image inference."""

    def __init__(
        self,
        checkpoint_path: Path | None = None,
        image_size: int | None = None,
        device: str | None = None,
        settings: AppSettings | None = None,
    ) -> None:
        self.settings = settings or SETTINGS
        self.checkpoint_path = checkpoint_path or self.settings.model_checkpoint_path
        self.image_size = image_size or self.settings.model_image_size
        configured_device = device or self.settings.resolve_model_device(
            torch.cuda.is_available()
        )
        self.device = torch.device(configured_device)
        self.model: nn.Module | None = None
        self.idx_to_class: dict[int, str] = {}
        self.class_names: list[str] = []

    def load(self) -> None:
        """Load model weights and class mapping from disk."""

        if self.checkpoint_path is None:
            raise FileNotFoundError("MODEL_CHECKPOINT_PATH is not configured.")
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Model checkpoint file does not exist: {self.checkpoint_path}"
            )

        self.class_names = self._load_class_names()
        self.idx_to_class = {
            index: class_name for index, class_name in enumerate(self.class_names)
        }

        self.model = self._build_model(num_classes=len(self.class_names)).to(self.device)

        checkpoint = self._torch_load_checkpoint(self.checkpoint_path)
        state_dict = self._extract_state_dict(checkpoint)
        self.model.load_state_dict(state_dict, strict=True)
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image: Image.Image,
        top_k: int = 3,
    ) -> tuple[PredictionResponse, list[PredictionResponse]]:
        """Predict the primary class plus the top-k ranked predictions."""

        if self.model is None:
            raise VisionModelUnavailableError(
                "Vision model is unavailable. Check MODEL_CHECKPOINT_PATH and "
                "MODEL_CLASSES_PATH."
            )
        if not self.idx_to_class:
            raise VisionModelUnavailableError("Class mapping is missing from checkpoint.")

        batch = preprocess_pil_image(
            image=image,
            image_size=self.image_size,
            device=self.device,
        )
        logits = self.model(batch)
        probabilities = torch.softmax(logits, dim=1)
        top_k = max(1, min(top_k, probabilities.shape[1]))
        top_confidences, top_indices = torch.topk(probabilities, k=top_k, dim=1)

        top_predictions: list[PredictionResponse] = []
        for confidence_tensor, index_tensor in zip(
            top_confidences[0],
            top_indices[0],
            strict=False,
        ):
            class_index = int(index_tensor.item())
            label = self.idx_to_class.get(class_index)
            if label is None:
                raise VisionModelUnavailableError(
                    f"Predicted class index is missing from mapping: {class_index}."
                )
            car = parse_class_label(label)
            top_predictions.append(
                PredictionResponse(
                    make=car.make,
                    model=car.model,
                    year=car.year,
                    class_label=label,
                    confidence=float(confidence_tensor.item()),
                )
            )

        return top_predictions[0], top_predictions

    def _build_model(self, num_classes: int) -> nn.Module:
        """Build the exact ResNet50 architecture used by the training script."""

        if num_classes != self.settings.model_num_classes:
            raise ValueError(
                f"classes.txt contains {num_classes} classes, but MODEL_NUM_CLASSES is "
                f"{self.settings.model_num_classes}."
            )

        model = models.resnet50(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=self.settings.model_dropout),
            nn.Linear(in_features, num_classes),
        )
        return model

    def _extract_state_dict(self, checkpoint: Any) -> dict[str, torch.Tensor]:
        """Support both raw state_dict files and wrapped checkpoint dictionaries."""

        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must be a state_dict or checkpoint dictionary.")

        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            if not isinstance(state_dict, dict):
                raise ValueError("'model_state_dict' must be a dictionary.")
            return state_dict

        return checkpoint

    def _load_class_names(self) -> list[str]:
        """Load class names from classes.txt in alphabetical order."""

        classes_path = self.settings.model_classes_path
        if not classes_path.is_file():
            raise FileNotFoundError(f"Class names file does not exist: {classes_path}")

        class_names = [
            line.strip()
            for line in classes_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not class_names:
            raise ValueError(f"Class names file is empty: {classes_path}")
        return class_names

    def _torch_load_checkpoint(self, checkpoint_path: Path) -> dict[str, Any]:
        """Load a checkpoint with the safest supported torch.load signature."""

        try:
            return torch.load(
                checkpoint_path,
                map_location=self.device,
                weights_only=True,
            )
        except TypeError:
            return torch.load(checkpoint_path, map_location=self.device)


feature_service = MockFeatureService()
valuation_service = MockValuationService()
vision_service = VisionInferenceService()

app = FastAPI(
    title="Car Vision Analysis API",
    version="1.0.0",
    description="Predicts car Make/Model/Year and returns mock features and valuations.",
)

if SETTINGS.cors_allowed_origin_values:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=list(SETTINGS.cors_allowed_origin_values),
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["*"],
    )

if SETTINGS.static_dir.exists():
    app.mount(
        "/static",
        StaticFiles(directory=str(SETTINGS.static_dir)),
        name="static",
    )


@app.on_event("startup")
def load_model_on_startup() -> None:
    """Load the model once when the API starts."""

    try:
        vision_service.load()
    except Exception:
        LOGGER.exception("Failed to load vision model.")
        raise


@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    """Serve the main web UI."""

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/health")
def health_check() -> dict[str, Any]:
    """Return API and model availability status."""

    return {
        "status": "ok",
        "model_loaded": vision_service.model is not None,
        "classes_loaded": len(vision_service.class_names),
        "checkpoint_path": (
            str(vision_service.checkpoint_path)
            if vision_service.checkpoint_path is not None
            else None
        ),
        "classes_path": str(SETTINGS.model_classes_path),
    }


@app.post("/analyze-car", response_model=AnalyzeCarResponse)
async def analyze_car(file: UploadFile = File(...)) -> AnalyzeCarResponse:
    """Analyze a car image and enrich the model prediction with mock data."""

    validate_upload(file, SETTINGS)
    image = await read_image(file, SETTINGS)

    try:
        prediction, top_predictions = vision_service.predict(image, top_k=3)
        car = CarIdentity(
            make=prediction.make,
            model=prediction.model,
            year=prediction.year,
        )
        features = feature_service.get_features(car)
        valuation = valuation_service.get_valuation(car)
    except VisionModelUnavailableError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc

    return AnalyzeCarResponse(
        prediction=prediction,
        top_predictions=top_predictions,
        features=features,
        valuation=ValuationResponse(**valuation.__dict__),
    )


def validate_upload(file: UploadFile, settings: AppSettings) -> None:
    """Validate uploaded file metadata before loading image bytes."""

    if (
        file.content_type is None
        or file.content_type not in settings.allowed_image_content_type_values
    ):
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                "Uploaded file must be one of: "
                f"{', '.join(settings.allowed_image_content_type_values)}."
            ),
        )


async def read_image(file: UploadFile, settings: AppSettings) -> Image.Image:
    """Read and decode an uploaded image."""

    try:
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Uploaded image is empty.",
            )
        if len(contents) > settings.max_upload_size_bytes:
            raise HTTPException(
                status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(
                    "Uploaded image exceeds the maximum size of "
                    f"{settings.max_upload_size_mb} MB."
                ),
            )
        return Image.open(io.BytesIO(contents)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Uploaded file could not be decoded as an image.",
        ) from exc
