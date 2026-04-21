"""ConvNeXt-Small + feature fusion model for KL grade classification (Path B)."""

import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig


class HybridKLClassifier(nn.Module):
    """Hybrid deep learning classifier combining ConvNeXt image features
    with extracted quantification features.

    Image encoder: ConvNeXt-Small (pretrained) -> backbone-dependent embedding
    Feature encoder: MLP on 50-dim extracted features -> 128-dim
    Fusion: concatenation -> MLP -> 5-class KL grade output
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()

        # Image branch: ConvNeXt-Small
        self.image_encoder = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            in_chans=cfg.in_channels,
        )
        img_dim = self.image_encoder.num_features

        # Enable gradient checkpointing for memory savings on M4
        if cfg.get("gradient_checkpointing", False):
            self.image_encoder.set_grad_checkpointing(True)

        # Feature branch: MLP on extracted features
        feature_dim = cfg.feature_dim
        self.feature_encoder = nn.Sequential(
            nn.Linear(feature_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_feature),
            nn.Linear(128, 128),
        )

        # Fusion + classification
        fusion_dim = img_dim + 128
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, cfg.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_fusion),
            nn.Linear(cfg.fusion_hidden_dim, cfg.num_classes),
        )

    def forward(
        self,
        image: torch.Tensor,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass with dual inputs.

        Args:
            image: (B, 1, 224, 224) full knee X-ray image.
            features: (B, 50) extracted feature vector.

        Returns:
            (B, 5) logits for KL grades 0-4.
        """
        img_feat = self.image_encoder(image)
        ext_feat = self.feature_encoder(features)
        fused = torch.cat([img_feat, ext_feat], dim=1)
        return self.classifier(fused)

    def freeze_backbone(self):
        """Freeze ConvNeXt backbone for initial training phase."""
        for param in self.image_encoder.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        """Unfreeze ConvNeXt backbone for fine-tuning phase."""
        for param in self.image_encoder.parameters():
            param.requires_grad = True
