"""Production-ready training script for the Turkey-focused car body type model.

Key features:
- Reads 105 classes from ``car_vision_project/data``.
- Skips corrupted or unreadable images safely.
- Uses a stratified 80/20 train/validation split.
- Applies anti-noise augmentation and label smoothing.
- Trains a pre-trained ResNet50 in two phases:
  1. Freeze backbone, train head for 5 epochs at 1e-3.
  2. Unfreeze all layers, fine-tune for 20 epochs at 1e-5.
- Saves:
  - ``best_car_model.pth``
  - ``classes.txt`` (alphabetical class names)
  - ``training_curves.png``
  - ``confusion_matrix.png``
  - ``training_summary.json``

Run from the repository root:

    python -m car_vision_project.train
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import random
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


LOGGER = logging.getLogger(__name__)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass(frozen=True)
class TrainingPhase:
    """Configuration for one training phase."""

    name: str
    epochs: int
    learning_rate: float
    freeze_backbone: bool


@dataclass(frozen=True)
class TrainingConfig:
    """Training configuration for the full pipeline."""

    data_dir: Path
    output_dir: Path
    expected_num_classes: int = 105
    image_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    validation_size: float = 0.20
    seed: int = 42
    label_smoothing: float = 0.1
    dropout: float = 0.3
    weight_decay: float = 1e-4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    phase_1_epochs: int = 5
    phase_1_lr: float = 1e-3
    phase_2_epochs: int = 20
    phase_2_lr: float = 1e-5

    def __post_init__(self) -> None:
        if self.expected_num_classes <= 1:
            raise ValueError("expected_num_classes must be greater than 1.")
        if self.image_size <= 0:
            raise ValueError("image_size must be greater than zero.")
        if self.batch_size <= 0:
            raise ValueError("batch_size must be greater than zero.")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative.")
        if not 0.0 < self.validation_size < 1.0:
            raise ValueError("validation_size must be in the range (0, 1).")
        if not 0.0 <= self.label_smoothing < 1.0:
            raise ValueError("label_smoothing must be in the range [0, 1).")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1).")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative.")
        if self.phase_1_epochs <= 0 or self.phase_2_epochs <= 0:
            raise ValueError("Both training phases must have at least one epoch.")
        if self.phase_1_lr <= 0 or self.phase_2_lr <= 0:
            raise ValueError("Learning rates must be greater than zero.")


@dataclass(frozen=True)
class SampleRecord:
    """One labeled image sample."""

    path: Path
    label: int


@dataclass(frozen=True)
class EpochMetrics:
    """Metrics captured after an epoch."""

    loss: float
    accuracy: float


class RobustCarDataset(Dataset[tuple[torch.Tensor, int] | None]):
    """Dataset that skips unreadable images instead of crashing training."""

    def __init__(
        self,
        samples: Sequence[SampleRecord],
        transform: transforms.Compose,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform
        self._warned_paths: set[Path] = set()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int] | None:
        sample = self.samples[index]
        try:
            with Image.open(sample.path) as image:
                rgb_image = image.convert("RGB")
            tensor = self.transform(rgb_image)
            return tensor, sample.label
        except (OSError, UnidentifiedImageError, ValueError) as exc:
            if sample.path not in self._warned_paths:
                LOGGER.warning("Skipping unreadable image '%s': %s", sample.path, exc)
                self._warned_paths.add(sample.path)
            return None


def safe_collate_fn(
    batch: list[tuple[torch.Tensor, int] | None],
) -> tuple[torch.Tensor, torch.Tensor] | None:
    """Filter out broken samples that returned ``None`` from the dataset."""

    valid_batch = [item for item in batch if item is not None]
    if not valid_batch:
        return None

    images, labels = zip(*valid_batch)
    return torch.stack(list(images)), torch.tensor(labels, dtype=torch.long)


def build_train_transform(image_size: int) -> transforms.Compose:
    """Training transform with anti-noise augmentations."""

    return transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size, scale=(0.65, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def build_val_transform(image_size: int) -> transforms.Compose:
    """Validation transform with deterministic resizing and normalization."""

    resize_size = int(round(image_size * 1.14))
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def discover_classes(data_dir: Path) -> list[str]:
    """Discover class folders in alphabetical order."""

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    if not data_dir.is_dir():
        raise NotADirectoryError(f"Data directory is not a directory: {data_dir}")

    class_dirs = sorted(
        path
        for path in data_dir.iterdir()
        if path.is_dir() and not path.name.startswith(".") and not path.name.startswith("__")
    )
    if not class_dirs:
        raise ValueError(f"No class directories found under: {data_dir}")

    empty_dirs = [path.name for path in class_dirs if not any(path.iterdir())]
    if empty_dirs:
        raise ValueError(f"Found empty class directories: {empty_dirs}")

    return [path.name for path in class_dirs]


def collect_samples(data_dir: Path, class_names: Sequence[str]) -> list[SampleRecord]:
    """Collect labeled image paths from class folders."""

    samples: list[SampleRecord] = []
    for label, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        files = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not files:
            raise ValueError(f"No supported image files found in class folder: {class_dir}")

        for path in files:
            samples.append(SampleRecord(path=path, label=label))

    if not samples:
        raise ValueError(f"No image samples found under: {data_dir}")
    return samples


def stratified_split(
    samples: Sequence[SampleRecord],
    validation_size: float,
    seed: int,
) -> tuple[list[SampleRecord], list[SampleRecord]]:
    """Perform a stratified train/validation split."""

    labels = [sample.label for sample in samples]
    label_counts = Counter(labels)
    small_classes = [label for label, count in label_counts.items() if count < 2]
    if small_classes:
        raise ValueError(
            "Each class must contain at least 2 images for stratified splitting. "
            f"Classes with insufficient samples: {small_classes}"
        )
    validation_sample_count = max(1, int(round(len(samples) * validation_size)))
    if validation_sample_count < len(label_counts):
        raise ValueError(
            "Validation split is too small for stratification across all classes. "
            f"Need at least {len(label_counts)} validation samples, "
            f"but computed {validation_sample_count}."
        )

    indices = np.arange(len(samples))
    train_indices, val_indices = train_test_split(
        indices,
        test_size=validation_size,
        random_state=seed,
        shuffle=True,
        stratify=labels,
    )

    train_samples = [samples[index] for index in train_indices]
    val_samples = [samples[index] for index in val_indices]
    return train_samples, val_samples


def create_dataloaders(
    train_samples: Sequence[SampleRecord],
    val_samples: Sequence[SampleRecord],
    config: TrainingConfig,
) -> tuple[DataLoader, DataLoader]:
    """Create train and validation dataloaders."""

    train_dataset = RobustCarDataset(
        samples=train_samples,
        transform=build_train_transform(config.image_size),
    )
    val_dataset = RobustCarDataset(
        samples=val_samples,
        transform=build_val_transform(config.image_size),
    )

    common_loader_args = {
        "batch_size": config.batch_size,
        "num_workers": config.num_workers,
        "pin_memory": config.device.startswith("cuda"),
        "collate_fn": safe_collate_fn,
        "persistent_workers": config.num_workers > 0,
    }

    train_loader = DataLoader(train_dataset, shuffle=True, **common_loader_args)
    val_loader = DataLoader(val_dataset, shuffle=False, **common_loader_args)
    return train_loader, val_loader


def build_model(num_classes: int, dropout: float) -> nn.Module:
    """Build a pre-trained ResNet50 with the requested classifier head."""

    weights = models.ResNet50_Weights.DEFAULT
    model = models.resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Dropout(p=dropout),
        nn.Linear(in_features, num_classes),
    )
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze or unfreeze the ResNet backbone while keeping the head trainable."""

    for name, parameter in model.named_parameters():
        if name.startswith("fc."):
            parameter.requires_grad = True
        else:
            parameter.requires_grad = trainable


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> int:
    """Return number of correct predictions in a batch."""

    predictions = torch.argmax(logits, dim=1)
    return int((predictions == labels).sum().item())


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    phase_name: str,
    epoch_label: str,
) -> EpochMetrics:
    """Run one train or validation epoch."""

    is_training = optimizer is not None
    model.train(mode=is_training)

    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    skipped_batches = 0

    from tqdm.auto import tqdm

    progress = tqdm(
        dataloader,
        desc=f"{phase_name} | {epoch_label} | {'train' if is_training else 'val'}",
        leave=False,
    )
    for batch in progress:
        if batch is None:
            skipped_batches += 1
            continue

        images, labels = batch
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)

        with torch.cuda.amp.autocast(enabled=device.type == "cuda"):
            logits = model(images)
            loss = criterion(logits, labels)

        if optimizer is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += accuracy_from_logits(logits, labels)
        total_seen += batch_size

        if total_seen > 0:
            progress.set_postfix(
                loss=f"{total_loss / total_seen:.4f}",
                acc=f"{total_correct / total_seen:.4f}",
                skipped=skipped_batches,
            )

    if total_seen == 0:
        raise RuntimeError(
            "No valid images were available in this epoch. "
            "Check dataset integrity and supported file formats."
        )

    return EpochMetrics(
        loss=total_loss / total_seen,
        accuracy=total_correct / total_seen,
    )


@torch.no_grad()
def collect_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[int]]:
    """Collect validation labels and predictions for the confusion matrix."""

    model.eval()
    targets: list[int] = []
    predictions: list[int] = []

    for batch in dataloader:
        if batch is None:
            continue

        images, labels = batch
        images = images.to(device, non_blocking=True)
        logits = model(images)
        predicted = torch.argmax(logits, dim=1).cpu().tolist()

        predictions.extend(predicted)
        targets.extend(labels.cpu().tolist())

    if not targets:
        raise RuntimeError("No validation predictions were collected.")

    return targets, predictions


def save_classes_txt(class_names: Sequence[str], output_path: Path) -> None:
    """Save Flask-compatible class names in alphabetical order."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(class_names) + "\n", encoding="utf-8")


def save_best_model(model: nn.Module, output_path: Path) -> None:
    """Save the model weights for Flask integration."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)


def save_training_summary(
    config: TrainingConfig,
    class_names: Sequence[str],
    history: list[dict[str, Any]],
    output_path: Path,
    best_val_accuracy: float,
) -> None:
    """Persist run metadata and epoch history."""

    summary = {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "num_classes": len(class_names),
        "classes": list(class_names),
        "best_val_accuracy": best_val_accuracy,
        "history": history,
    }
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")


def plot_training_curves(history: Sequence[dict[str, Any]], output_path: Path) -> None:
    """Plot training and validation accuracy/loss curves."""

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib is required to plot training curves. Install it first."
        ) from exc

    epochs = [entry["epoch"] for entry in history]
    train_loss = [entry["train_loss"] for entry in history]
    val_loss = [entry["val_loss"] for entry in history]
    train_acc = [entry["train_accuracy"] for entry in history]
    val_acc = [entry["val_accuracy"] for entry in history]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_loss, label="Train Loss", linewidth=2)
    axes[0].plot(epochs, val_loss, label="Val Loss", linewidth=2)
    axes[0].set_title("Loss Curves")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(epochs, train_acc, label="Train Accuracy", linewidth=2)
    axes[1].plot(epochs, val_acc, label="Val Accuracy", linewidth=2)
    axes[1].set_title("Accuracy Curves")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_confusion_matrix(
    targets: Sequence[int],
    predictions: Sequence[int],
    class_names: Sequence[str],
    output_path: Path,
) -> None:
    """Plot a Seaborn confusion matrix for validation predictions."""

    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError as exc:
        raise RuntimeError(
            "matplotlib and seaborn are required to plot the confusion matrix."
        ) from exc

    matrix = confusion_matrix(targets, predictions, labels=list(range(len(class_names))))
    figure_size = max(18, len(class_names) * 0.28)

    fig, ax = plt.subplots(figsize=(figure_size, figure_size))
    sns.heatmap(
        matrix,
        cmap="Blues",
        cbar=True,
        square=True,
        xticklabels=class_names,
        yticklabels=class_names,
        ax=ax,
    )
    ax.set_title("Validation Confusion Matrix")
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")
    ax.tick_params(axis="x", labelrotation=90, labelsize=7)
    ax.tick_params(axis="y", labelrotation=0, labelsize=7)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def seed_everything(seed: int) -> None:
    """Set deterministic seeds for reproducibility."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True


def build_phases(config: TrainingConfig) -> list[TrainingPhase]:
    """Create the two mandated training phases."""

    return [
        TrainingPhase(
            name="phase_1_head_training",
            epochs=config.phase_1_epochs,
            learning_rate=config.phase_1_lr,
            freeze_backbone=True,
        ),
        TrainingPhase(
            name="phase_2_fine_tuning",
            epochs=config.phase_2_epochs,
            learning_rate=config.phase_2_lr,
            freeze_backbone=False,
        ),
    ]


def train(config: TrainingConfig) -> None:
    """Train the model end-to-end and export artifacts."""

    seed_everything(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    class_names = discover_classes(config.data_dir)
    if len(class_names) != config.expected_num_classes:
        raise ValueError(
            f"Expected exactly {config.expected_num_classes} classes, "
            f"but found {len(class_names)}."
        )

    samples = collect_samples(config.data_dir, class_names)
    train_samples, val_samples = stratified_split(
        samples=samples,
        validation_size=config.validation_size,
        seed=config.seed,
    )
    train_loader, val_loader = create_dataloaders(train_samples, val_samples, config)

    save_classes_txt(class_names, config.output_dir / "classes.txt")

    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available.")

    device = torch.device(config.device)
    model = build_model(num_classes=len(class_names), dropout=config.dropout).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_val_accuracy = 0.0
    best_state_dict = copy.deepcopy(model.state_dict())
    history: list[dict[str, Any]] = []

    LOGGER.info("Classes discovered: %d", len(class_names))
    LOGGER.info("Samples collected: %d", len(samples))
    LOGGER.info("Train samples: %d | Val samples: %d", len(train_samples), len(val_samples))
    LOGGER.info("Device: %s", device)

    global_epoch = 0
    for phase in build_phases(config):
        set_backbone_trainable(model, trainable=not phase.freeze_backbone)
        optimizer = Adam(
            params=[parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=phase.learning_rate,
            weight_decay=config.weight_decay,
        )

        LOGGER.info(
            "Starting %s | epochs=%d | lr=%g | freeze_backbone=%s",
            phase.name,
            phase.epochs,
            phase.learning_rate,
            phase.freeze_backbone,
        )

        for phase_epoch in range(1, phase.epochs + 1):
            global_epoch += 1
            epoch_label = f"epoch {global_epoch}"

            train_metrics = run_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                scaler=scaler,
                phase_name=phase.name,
                epoch_label=epoch_label,
            )
            val_metrics = run_epoch(
                model=model,
                dataloader=val_loader,
                criterion=criterion,
                optimizer=None,
                device=device,
                scaler=scaler,
                phase_name=phase.name,
                epoch_label=epoch_label,
            )

            history_entry = {
                "epoch": global_epoch,
                "phase": phase.name,
                "phase_epoch": phase_epoch,
                "train_loss": train_metrics.loss,
                "train_accuracy": train_metrics.accuracy,
                "val_loss": val_metrics.loss,
                "val_accuracy": val_metrics.accuracy,
            }
            history.append(history_entry)

            LOGGER.info(
                "%s | train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
                epoch_label,
                train_metrics.loss,
                train_metrics.accuracy,
                val_metrics.loss,
                val_metrics.accuracy,
            )

            if val_metrics.accuracy > best_val_accuracy:
                best_val_accuracy = val_metrics.accuracy
                best_state_dict = copy.deepcopy(model.state_dict())
                save_best_model(model, config.output_dir / "best_car_model.pth")
                LOGGER.info(
                    "New best model saved with validation accuracy %.4f",
                    best_val_accuracy,
                )

    model.load_state_dict(best_state_dict)
    targets, predictions = collect_predictions(model, val_loader, device)

    plot_training_curves(history, config.output_dir / "training_curves.png")
    plot_confusion_matrix(
        targets=targets,
        predictions=predictions,
        class_names=class_names,
        output_path=config.output_dir / "confusion_matrix.png",
    )
    save_training_summary(
        config=config,
        class_names=class_names,
        history=history,
        output_path=config.output_dir / "training_summary.json",
        best_val_accuracy=best_val_accuracy,
    )

    LOGGER.info("Training complete. Best validation accuracy: %.4f", best_val_accuracy)
    LOGGER.info("Artifacts saved in: %s", config.output_dir)


def parse_args() -> TrainingConfig:
    """Parse CLI arguments."""

    project_root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Train a 105-class car body type classifier.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=project_root / "data",
        help="Directory containing 105 class subfolders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=project_root / "artifacts",
        help="Directory where weights, classes.txt, and plots are saved.",
    )
    parser.add_argument("--expected-num-classes", type=int, default=105)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=min(4, os.cpu_count() or 1))
    parser.add_argument("--validation-size", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    parser.add_argument("--phase-1-epochs", type=int, default=5)
    parser.add_argument("--phase-1-lr", type=float, default=1e-3)
    parser.add_argument("--phase-2-epochs", type=int, default=20)
    parser.add_argument("--phase-2-lr", type=float, default=1e-5)

    args = parser.parse_args()
    return TrainingConfig(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        expected_num_classes=args.expected_num_classes,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        validation_size=args.validation_size,
        seed=args.seed,
        label_smoothing=args.label_smoothing,
        dropout=args.dropout,
        weight_decay=args.weight_decay,
        device=args.device,
        phase_1_epochs=args.phase_1_epochs,
        phase_1_lr=args.phase_1_lr,
        phase_2_epochs=args.phase_2_epochs,
        phase_2_lr=args.phase_2_lr,
    )


def main() -> None:
    """CLI entrypoint."""

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    try:
        train(parse_args())
    except Exception:
        LOGGER.exception("Training failed.")
        raise


if __name__ == "__main__":
    main()
