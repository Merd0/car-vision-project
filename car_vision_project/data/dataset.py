"""Dataset and DataLoader builders for car image classification."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import datasets

from car_vision_project.utils.image_transforms import (
    ImageTransformConfig,
    ImageTransformFactory,
)


@dataclass(frozen=True)
class CarImageDatasetConfig:
    """Configuration for an ImageFolder-based car dataset."""

    data_dir: Path
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = False

    def __post_init__(self) -> None:
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")


class CarImageDataModule:
    """Build training and validation datasets from an ImageFolder layout."""

    def __init__(self, config: CarImageDatasetConfig) -> None:
        self.config = config
        self.transform_factory = ImageTransformFactory(
            ImageTransformConfig(image_size=config.image_size)
        )

    def build_dataloaders(self) -> tuple[DataLoader, DataLoader, dict[str, int]]:
        """Return train loader, validation loader, and class mapping."""

        train_dataset, val_dataset = self.build_datasets()
        self._validate_class_mapping(train_dataset.class_to_idx, val_dataset.class_to_idx)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=self.config.num_workers,
            pin_memory=self.config.pin_memory,
        )
        return train_loader, val_loader, train_dataset.class_to_idx

    def build_datasets(self) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
        """Create ImageFolder datasets for ``train`` and ``val`` splits."""

        train_dir = self.config.data_dir / "train"
        val_dir = self.config.data_dir / "val"
        self.validate_split_dir(train_dir)
        self.validate_split_dir(val_dir)

        train_dataset = datasets.ImageFolder(
            root=train_dir,
            transform=self.transform_factory.build_train_transform(),
        )
        val_dataset = datasets.ImageFolder(
            root=val_dir,
            transform=self.transform_factory.build_eval_transform(),
        )

        if len(train_dataset.classes) < 2:
            raise ValueError("Training requires at least two classes.")
        return train_dataset, val_dataset

    @staticmethod
    def validate_split_dir(path: Path) -> None:
        """Validate that an ImageFolder split exists and contains class folders."""

        if not path.exists():
            raise FileNotFoundError(f"Dataset split does not exist: {path}")
        if not path.is_dir():
            raise NotADirectoryError(f"Dataset split is not a directory: {path}")
        if not any(child.is_dir() for child in path.iterdir()):
            raise ValueError(f"Dataset split contains no class folders: {path}")

    @staticmethod
    def _validate_class_mapping(
        train_class_to_idx: dict[str, int],
        val_class_to_idx: dict[str, int],
    ) -> None:
        if train_class_to_idx != val_class_to_idx:
            raise ValueError(
                "Train and validation class mappings differ. Ensure both splits "
                "contain the same class folder names."
            )
