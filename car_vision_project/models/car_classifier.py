"""Configurable car image classifier built on torchvision backbones.

The model predicts a single class representing a normalized
Make/Model/Year label, such as ``toyota_corolla_2020``. Downstream API
code can map that class label into structured fields before calling the
feature and valuation services.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import nn
from torchvision import models


@dataclass(frozen=True)
class BackboneSpec:
    """Factory metadata for a supported torchvision backbone."""

    factory: Callable[..., nn.Module]
    weights: Any
    classifier_attr: str
    in_features_attr: str


SUPPORTED_BACKBONES: dict[str, BackboneSpec] = {
    "resnet18": BackboneSpec(
        factory=models.resnet18,
        weights=models.ResNet18_Weights.DEFAULT,
        classifier_attr="fc",
        in_features_attr="in_features",
    ),
    "resnet34": BackboneSpec(
        factory=models.resnet34,
        weights=models.ResNet34_Weights.DEFAULT,
        classifier_attr="fc",
        in_features_attr="in_features",
    ),
    "resnet50": BackboneSpec(
        factory=models.resnet50,
        weights=models.ResNet50_Weights.DEFAULT,
        classifier_attr="fc",
        in_features_attr="in_features",
    ),
    "efficientnet_b0": BackboneSpec(
        factory=models.efficientnet_b0,
        weights=models.EfficientNet_B0_Weights.DEFAULT,
        classifier_attr="classifier",
        in_features_attr="in_features",
    ),
    "efficientnet_b1": BackboneSpec(
        factory=models.efficientnet_b1,
        weights=models.EfficientNet_B1_Weights.DEFAULT,
        classifier_attr="classifier",
        in_features_attr="in_features",
    ),
    "efficientnet_b2": BackboneSpec(
        factory=models.efficientnet_b2,
        weights=models.EfficientNet_B2_Weights.DEFAULT,
        classifier_attr="classifier",
        in_features_attr="in_features",
    ),
}


class CarClassifier(nn.Module):
    """Transfer-learning classifier for car Make/Model/Year recognition.

    Parameters
    ----------
    num_classes:
        Number of target classes in the training dataset.
    backbone_name:
        Torchvision backbone name. Supported values are defined in
        ``SUPPORTED_BACKBONES``.
    pretrained:
        If ``True``, initialize the backbone with ImageNet weights.
    dropout:
        Dropout probability used in the replacement classifier head.
    freeze_backbone:
        If ``True``, freeze all backbone parameters except the new head.
    hidden_dim:
        Optional hidden layer size for the classifier head. If omitted,
        a single linear classification layer is used.
    """

    def __init__(
        self,
        num_classes: int,
        backbone_name: str = "efficientnet_b0",
        pretrained: bool = True,
        dropout: float = 0.2,
        freeze_backbone: bool = False,
        hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("num_classes must be greater than zero.")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0.0, 1.0).")
        if hidden_dim is not None and hidden_dim <= 0:
            raise ValueError("hidden_dim must be greater than zero when provided.")

        self.num_classes = num_classes
        self.backbone_name = backbone_name
        self.pretrained = pretrained
        self.dropout = dropout
        self.hidden_dim = hidden_dim
        self.freeze_backbone_enabled = freeze_backbone

        spec = self._get_backbone_spec(backbone_name)
        weights = spec.weights if pretrained else None
        self.backbone = spec.factory(weights=weights)

        in_features = self._replace_classifier_head(spec)
        self.in_features = in_features

        if freeze_backbone:
            self.freeze_backbone()
            self.unfreeze_classifier()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return unnormalized class logits for a batch of images."""

        return self.backbone(images)

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint_path: str | Path,
        map_location: str | torch.device = "cpu",
    ) -> "CarClassifier":
        """Build a model and load weights from a checkpoint saved by this class."""

        checkpoint = torch.load(checkpoint_path, map_location=map_location)
        if not isinstance(checkpoint, Mapping):
            raise ValueError("Checkpoint must contain a mapping object.")

        model_config = checkpoint.get("model_config")
        state_dict = checkpoint.get("model_state_dict")
        if not isinstance(model_config, Mapping) or state_dict is None:
            raise ValueError(
                "Checkpoint must include 'model_config' and 'model_state_dict'."
            )

        model = cls(**dict(model_config))
        model.load_state_dict(state_dict)
        return model

    def save_checkpoint(
        self,
        checkpoint_path: str | Path,
        class_to_idx: Mapping[str, int] | None = None,
        extra_metadata: Mapping[str, Any] | None = None,
    ) -> None:
        """Persist model weights and enough metadata to recreate the model."""

        path = Path(checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint: dict[str, Any] = {
            "model_config": self.model_config,
            "model_state_dict": self.state_dict(),
        }
        if class_to_idx is not None:
            checkpoint["class_to_idx"] = dict(class_to_idx)
        if extra_metadata is not None:
            checkpoint["metadata"] = dict(extra_metadata)

        torch.save(checkpoint, path)

    def freeze_backbone(self) -> None:
        """Freeze every parameter in the underlying backbone."""

        for parameter in self.backbone.parameters():
            parameter.requires_grad = False

    def unfreeze_backbone(self) -> None:
        """Unfreeze every parameter in the underlying backbone."""

        for parameter in self.backbone.parameters():
            parameter.requires_grad = True

    def unfreeze_classifier(self) -> None:
        """Unfreeze only the replacement classifier head."""

        for parameter in self._classifier_module.parameters():
            parameter.requires_grad = True

    @property
    def model_config(self) -> dict[str, Any]:
        """Return serializable constructor arguments for checkpointing."""

        return {
            "num_classes": self.num_classes,
            "backbone_name": self.backbone_name,
            "pretrained": self.pretrained,
            "dropout": self.dropout,
            "freeze_backbone": self.freeze_backbone_enabled,
            "hidden_dim": self.hidden_dim,
        }

    @property
    def _classifier_module(self) -> nn.Module:
        spec = self._get_backbone_spec(self.backbone_name)
        return getattr(self.backbone, spec.classifier_attr)

    @staticmethod
    def _get_backbone_spec(backbone_name: str) -> BackboneSpec:
        try:
            return SUPPORTED_BACKBONES[backbone_name]
        except KeyError as exc:
            supported = ", ".join(sorted(SUPPORTED_BACKBONES))
            raise ValueError(
                f"Unsupported backbone '{backbone_name}'. Supported: {supported}."
            ) from exc

    def _replace_classifier_head(self, spec: BackboneSpec) -> int:
        classifier = getattr(self.backbone, spec.classifier_attr)

        if isinstance(classifier, nn.Linear):
            in_features = getattr(classifier, spec.in_features_attr)
        elif isinstance(classifier, nn.Sequential) and len(classifier) > 0:
            last_linear = self._find_last_linear(classifier)
            in_features = getattr(last_linear, spec.in_features_attr)
        else:
            raise TypeError(
                f"Unsupported classifier module type: {type(classifier).__name__}."
            )

        new_head = self._build_classifier_head(in_features)
        setattr(self.backbone, spec.classifier_attr, new_head)
        return in_features

    def _build_classifier_head(self, in_features: int) -> nn.Module:
        if self.hidden_dim is None:
            return nn.Sequential(
                nn.Dropout(p=self.dropout),
                nn.Linear(in_features, self.num_classes),
            )

        return nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(in_features, self.hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(self.hidden_dim, self.num_classes),
        )

    @staticmethod
    def _find_last_linear(module: nn.Sequential) -> nn.Linear:
        for layer in reversed(module):
            if isinstance(layer, nn.Linear):
                return layer
        raise TypeError("Classifier sequence does not contain a linear layer.")
