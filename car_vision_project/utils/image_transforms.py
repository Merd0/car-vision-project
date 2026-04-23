"""Image preprocessing factories for training, validation, and inference."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from PIL import Image
from torchvision import transforms


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class ImageTransformConfig:
    """Configuration shared by all image transform pipelines."""

    image_size: int = 224
    normalize_mean: list[float] | None = None
    normalize_std: list[float] | None = None

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero.")

    @property
    def mean(self) -> list[float]:
        """Return normalization mean values."""

        return self.normalize_mean or IMAGENET_MEAN

    @property
    def std(self) -> list[float]:
        """Return normalization standard deviation values."""

        return self.normalize_std or IMAGENET_STD


class ImageTransformFactory:
    """Factory for consistent torchvision preprocessing pipelines."""

    def __init__(self, config: ImageTransformConfig | None = None) -> None:
        self.config = config or ImageTransformConfig()

    def build_train_transform(self) -> transforms.Compose:
        """Create augmenting transforms for model training."""

        return transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.mean, std=self.config.std),
            ]
        )

    def build_eval_transform(self) -> transforms.Compose:
        """Create deterministic transforms for validation and inference."""

        return transforms.Compose(
            [
                transforms.Resize((self.config.image_size, self.config.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.config.mean, std=self.config.std),
            ]
        )


def preprocess_pil_image(
    image: Image.Image,
    image_size: int = 224,
    device: torch.device | str = "cpu",
) -> torch.Tensor:
    """Convert a PIL image into a normalized batch tensor."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image instance.")

    transform = ImageTransformFactory(
        ImageTransformConfig(image_size=image_size)
    ).build_eval_transform()
    tensor = transform(image.convert("RGB")).unsqueeze(0)
    return tensor.to(device)
