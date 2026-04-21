"""Hybrid CNN + Texture MLP model for subchondral sclerosis classification."""

import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig


class SclerosisClassifier(nn.Module):
    """Hybrid model fusing CNN image features with handcrafted texture features.

    CNN branch: EfficientNet-B0 on subchondral ROI patches -> 1280-dim
    Texture branch: MLP on precomputed LBP/GLCM/FD/intensity features -> 64-dim
    Fusion: concatenation + MLP -> 3-class output (none/mild/significant)
    """

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.num_classes = int(cfg.num_classes)
        self.use_side_specific_heads = bool(getattr(cfg, "use_side_specific_heads", False))

        # CNN branch
        self.cnn = timm.create_model(
            cfg.cnn_backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            in_chans=cfg.in_channels,
        )
        cnn_dim = self.cnn.num_features  # 1280 for EfficientNet-B0

        # Texture MLP branch
        texture_dim = cfg.texture_feature_dim
        self.texture_mlp = nn.Sequential(
            nn.Linear(texture_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_cnn),
            nn.Linear(128, 64),
        )

        fusion_input_dim = cnn_dim + 64
        self.shared_fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, cfg.fusion_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.dropout_fusion),
        )

        if self.use_side_specific_heads:
            side_head_hidden_dim = int(getattr(cfg, "side_head_hidden_dim", cfg.fusion_hidden_dim))
            self.medial_head = nn.Sequential(
                nn.Linear(cfg.fusion_hidden_dim, side_head_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout_fusion),
                nn.Linear(side_head_hidden_dim, self.num_classes),
            )
            self.lateral_head = nn.Sequential(
                nn.Linear(cfg.fusion_hidden_dim, side_head_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout_fusion),
                nn.Linear(side_head_hidden_dim, self.num_classes),
            )
            self.side_embedding_dim = 0
            self.side_embedding = None
            self.classifier = None
        else:
            self.side_embedding_dim = int(getattr(cfg, "side_embedding_dim", 8))
            self.side_embedding = nn.Embedding(2, self.side_embedding_dim)
            self.classifier = nn.Sequential(
                nn.Linear(cfg.fusion_hidden_dim + self.side_embedding_dim, cfg.fusion_hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(cfg.dropout_fusion),
                nn.Linear(cfg.fusion_hidden_dim, self.num_classes),
            )
            self.medial_head = None
            self.lateral_head = None

    def forward(
        self,
        roi_image: torch.Tensor,
        texture_features: torch.Tensor,
        side_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass with dual inputs.

        Args:
            roi_image: (B, 1, H, W) subchondral ROI patch.
            texture_features: (B, D) precomputed texture feature vector.
            side_ids: (B,) compartment ids (0=medial, 1=lateral).

        Returns:
            (B, 3) logits for sclerosis grades.
        """
        cnn_feat = self.cnn(roi_image)
        tex_feat = self.texture_mlp(texture_features)
        fused = self.shared_fusion(torch.cat([cnn_feat, tex_feat], dim=1))
        if side_ids is None:
            side_ids = torch.zeros(roi_image.shape[0], dtype=torch.long, device=roi_image.device)

        if self.use_side_specific_heads:
            side_ids = side_ids.long()
            logits = torch.empty(
                (fused.shape[0], self.num_classes),
                dtype=fused.dtype,
                device=fused.device,
            )
            medial_mask = side_ids == 0
            lateral_mask = ~medial_mask
            if medial_mask.any():
                logits[medial_mask] = self.medial_head(fused[medial_mask])
            if lateral_mask.any():
                logits[lateral_mask] = self.lateral_head(fused[lateral_mask])
            return logits

        side_feat = self.side_embedding(side_ids.long())
        return self.classifier(torch.cat([fused, side_feat], dim=1))
