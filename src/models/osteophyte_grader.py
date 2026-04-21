"""SE-ResNet-50 multi-head model for per-ROI osteophyte grading."""

from typing import Dict, Tuple

import timm
import torch
import torch.nn as nn
from omegaconf import DictConfig


class OsteophyteGrader(nn.Module):
    """SE-ResNet-50 backbone with 4 classification heads for OARSI grading.

    Each head classifies osteophyte severity (0-3) at one anatomical site:
    medial femur, lateral femur, medial tibia, lateral tibia.
    """

    SITES = ["medial_femur", "lateral_femur", "medial_tibia", "lateral_tibia"]

    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.backbone = timm.create_model(
            cfg.backbone,
            pretrained=cfg.pretrained,
            num_classes=0,
            in_chans=cfg.in_channels,
            drop_rate=float(getattr(cfg, "dropout", 0.0)),
        )
        feature_dim = self.backbone.num_features  # 2048 for SE-ResNet-50
        head_hidden = int(getattr(cfg, "head_hidden_dim", 256))
        head_dropout_pre = float(getattr(cfg, "head_dropout_pre", 0.4))
        head_dropout_post = float(getattr(cfg, "head_dropout_post", 0.3))
        use_mlp_heads = bool(getattr(cfg, "use_mlp_heads", True))

        if use_mlp_heads:
            self.heads = nn.ModuleDict({
                site: nn.Sequential(
                    nn.Dropout(head_dropout_pre),
                    nn.Linear(feature_dim, head_hidden),
                    nn.ReLU(inplace=True),
                    nn.Dropout(head_dropout_post),
                    nn.Linear(head_hidden, cfg.num_classes_per_head),
                )
                for site in self.SITES
            })
        else:
            self.heads = nn.ModuleDict({
                site: nn.Linear(feature_dim, cfg.num_classes_per_head)
                for site in self.SITES
            })

    def forward_single(self, x: torch.Tensor, site: str) -> torch.Tensor:
        """Forward pass for a single ROI site."""
        features = self.backbone(x)
        return self.heads[site](features)

    def forward(
        self,
        x_mf: torch.Tensor,
        x_lf: torch.Tensor,
        x_mt: torch.Tensor,
        x_lt: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass for all 4 ROI sites.

        Args:
            x_mf, x_lf, x_mt, x_lt: (B, 1, H, W) ROI patch tensors.

        Returns:
            Dict mapping site name to (B, 4) logits.
        """
        # Shared backbone features
        f_mf = self.backbone(x_mf)
        f_lf = self.backbone(x_lf)
        f_mt = self.backbone(x_mt)
        f_lt = self.backbone(x_lt)

        return {
            "medial_femur": self.heads["medial_femur"](f_mf),
            "lateral_femur": self.heads["lateral_femur"](f_lf),
            "medial_tibia": self.heads["medial_tibia"](f_mt),
            "lateral_tibia": self.heads["lateral_tibia"](f_lt),
        }

    def extract_osteophyte_features(self, predictions: Dict[str, torch.Tensor]) -> Dict[str, int]:
        """Convert model predictions to the 10-dim osteophyte feature dict."""
        grades = {}
        for site in self.SITES:
            grade = predictions[site].argmax(dim=1).item()
            abbrev = {"medial_femur": "mf", "lateral_femur": "lf",
                      "medial_tibia": "mt", "lateral_tibia": "lt"}[site]
            grades[f"osp_grade_{abbrev}"] = grade

        mf = grades["osp_grade_mf"]
        lf = grades["osp_grade_lf"]
        mt = grades["osp_grade_mt"]
        lt = grades["osp_grade_lt"]

        grades["osp_sum"] = mf + lf + mt + lt
        grades["osp_max"] = max(mf, lf, mt, lt)
        grades["osp_medial_sum"] = mf + mt
        grades["osp_lateral_sum"] = lf + lt
        grades["osp_femoral_sum"] = mf + lf
        grades["osp_tibial_sum"] = mt + lt

        return grades
