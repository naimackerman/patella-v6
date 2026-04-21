"""Helpers for working with PyTorch Lightning checkpoints."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence

import torch


def load_checkpoint(checkpoint_path: str | Path, map_location: str | torch.device = "cpu") -> Dict:
    """Load a Lightning checkpoint with PyTorch 2.6 compatibility."""
    return torch.load(Path(checkpoint_path), map_location=map_location, weights_only=False)


def extract_model_state_dict(checkpoint: Dict, prefix: str = "model.") -> Dict[str, torch.Tensor]:
    """Extract a model-only state dict from a Lightning checkpoint."""
    state_dict = checkpoint.get("state_dict", {})
    if not prefix:
        return state_dict
    return {
        key.replace(prefix, "", 1): value
        for key, value in state_dict.items()
        if key.startswith(prefix)
    }


def _normalize_monitor_name(name: Optional[str]) -> Optional[str]:
    if name is None:
        return None
    return str(name).replace("/", "_")


def checkpoint_score(checkpoint_path: str | Path, monitor: Optional[str] = None) -> float:
    """Read the stored monitor score from a Lightning checkpoint."""
    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    callbacks = checkpoint.get("callbacks", {})

    normalized_monitor = _normalize_monitor_name(monitor)
    fallback_score = None
    for callback_key, callback_state in callbacks.items():
        if not isinstance(callback_state, dict):
            continue
        callback_monitor = callback_state.get("monitor")
        normalized_callback_monitor = _normalize_monitor_name(callback_monitor)

        current_score = callback_state.get("current_score")
        best_score = callback_state.get("best_model_score")
        score = current_score if current_score is not None else best_score
        if score is None:
            continue
        score_value = float(score.item()) if hasattr(score, "item") else float(score)
        if normalized_monitor is not None and normalized_callback_monitor != normalized_monitor:
            if fallback_score is None:
                fallback_score = score_value
            continue
        return score_value

    if normalized_monitor is not None:
        return float("-inf")
    return fallback_score if fallback_score is not None else float("-inf")


def find_best_lightning_checkpoint(
    checkpoint_dir: str | Path,
    pattern: str = "*.ckpt",
    monitor: Optional[str] = None,
    mode: str = "max",
) -> Optional[Path]:
    """Return the checkpoint with the best stored monitor score."""
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_paths = sorted(checkpoint_dir.glob(pattern))
    if not checkpoint_paths:
        return None

    scored = [
        (checkpoint_score(checkpoint_path, monitor=monitor), checkpoint_path)
        for checkpoint_path in checkpoint_paths
    ]
    valid = [item for item in scored if item[0] != float("-inf")]
    if not valid:
        return None
    reverse = str(mode).lower() != "min"
    valid.sort(key=lambda item: item[0], reverse=reverse)
    return valid[0][1]


def resolve_osteophyte_checkpoint_paths(
    checkpoint_dir: str | Path,
    sites: Sequence[str],
    prefer_refined: bool = True,
    force_multitask_sites: Optional[Sequence[str]] = None,
    force_refined_sites: Optional[Sequence[str]] = None,
    override_paths_by_site: Optional[dict[str, str | Path]] = None,
) -> dict[str, dict[str, Path | str]]:
    """Resolve the best checkpoint for each osteophyte site.

    Resolution order per site:
      1. explicit per-site override when provided
      2. site-refined checkpoint (`osp-refined-{site}-*.ckpt`) when available
      3. multitask checkpoint (`osp-multitask-*.ckpt`) as the default fallback
      4. legacy single-site checkpoint (`osp-{site}-*.ckpt`) only when no multitask
         checkpoint is available
    """
    checkpoint_dir = Path(checkpoint_dir)
    resolved: dict[str, dict[str, Path | str]] = {}
    force_multitask_sites = {str(site_name) for site_name in (force_multitask_sites or [])}
    force_refined_sites = {str(site_name) for site_name in (force_refined_sites or [])}
    override_paths_by_site = {
        str(site_name): Path(path_value)
        for site_name, path_value in (override_paths_by_site or {}).items()
        if path_value is not None
    }
    multitask_ckpt = find_best_lightning_checkpoint(
        checkpoint_dir,
        pattern="osp-multitask-*.ckpt",
        monitor="val_kappa_mean",
    )

    for site in sites:
        override_ckpt = override_paths_by_site.get(site)
        if override_ckpt is not None:
            resolved[site] = {"mode": "override", "path": override_ckpt}
            continue
        current_prefer_refined = prefer_refined and site not in force_multitask_sites
        if site in force_refined_sites:
            current_prefer_refined = True
        if current_prefer_refined:
            refined_ckpt = find_best_lightning_checkpoint(
                checkpoint_dir,
                pattern=f"osp-refined-{site}-*.ckpt",
                monitor="val/kappa",
            )
            if refined_ckpt is not None:
                resolved[site] = {"mode": "refined_site", "path": refined_ckpt}
                continue

        if multitask_ckpt is not None:
            resolved[site] = {"mode": "multitask", "path": multitask_ckpt}
            continue

        single_site_ckpt = find_best_lightning_checkpoint(
            checkpoint_dir,
            pattern=f"osp-{site}-*.ckpt",
            monitor="val/kappa",
        )
        if single_site_ckpt is not None:
            resolved[site] = {"mode": "site", "path": single_site_ckpt}

    return resolved
