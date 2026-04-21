"""Utilities for configurable sclerosis label schemes."""

from __future__ import annotations

import numpy as np
from omegaconf import DictConfig, OmegaConf

SEVERITY_CLASS_NAMES = ["none", "mild", "significant"]
BINARY_PRESENT_CLASS_NAMES = ["none", "present"]


def normalize_sclerosis_label_scheme(scheme: str | None) -> str:
    value = str(scheme or "severity").strip().lower().replace("-", "_")
    aliases = {
        "3class": "severity",
        "3_class": "severity",
        "three_class": "severity",
        "ordinal": "severity",
        "severity_3class": "severity",
        "binary": "binary_present",
        "present": "binary_present",
        "none_vs_present": "binary_present",
        "any": "binary_present",
        "any_present": "binary_present",
    }
    value = aliases.get(value, value)
    if value not in {"severity", "binary_present"}:
        raise ValueError(
            f"Unsupported sclerosis_label_scheme={scheme!r}. "
            "Use 'severity' or 'binary_present'."
        )
    return value


def sclerosis_class_names(scheme: str | None) -> list[str]:
    scheme = normalize_sclerosis_label_scheme(scheme)
    if scheme == "binary_present":
        return BINARY_PRESENT_CLASS_NAMES.copy()
    return SEVERITY_CLASS_NAMES.copy()


def map_sclerosis_grades(grades: np.ndarray, scheme: str | None) -> np.ndarray:
    grades = np.asarray(grades, dtype=np.int64)
    scheme = normalize_sclerosis_label_scheme(scheme)
    if scheme == "binary_present":
        return (grades > 0).astype(np.int64)
    return grades


def apply_sclerosis_label_scheme_to_cfg(cfg: DictConfig, scheme: str | None = None) -> DictConfig:
    scheme = normalize_sclerosis_label_scheme(
        scheme if scheme is not None else getattr(cfg.training, "sclerosis_label_scheme", "severity")
    )
    names = sclerosis_class_names(scheme)
    return OmegaConf.merge(
        cfg,
        {
            "training": {"sclerosis_label_scheme": scheme},
            "model": {"num_classes": len(names), "class_names": names},
        },
    )
