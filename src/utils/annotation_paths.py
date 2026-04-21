"""Helpers for resolving annotation files and label subsets by study phase."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


MANUAL_SOURCES = {"manual_review", "reviewed_manual", "manual"}
EXPANDED_SOURCES = MANUAL_SOURCES | {"high_conf_model"}
BOOTSTRAP_SOURCES = {
    "heuristic_image_only",
    "bootstrap_rule",
    "bootstrap_pseudo",
    "model_prediction",
    "fallback_all",
    "fallback_all(auto)",
}


def normalize_label_mode(mode: str | None) -> str:
    """Normalize supported label mode aliases."""
    normalized = (mode or "manual").lower()
    if normalized == "pseudo":
        return "bootstrap"
    if normalized == "all":
        return "bootstrap"
    return normalized


def resolve_annotation_csv(
    annotation_dir: str | Path,
    stem: str,
    mode: str = "manual",
    allow_bootstrap_fallback: bool = False,
) -> Path:
    """Resolve the most appropriate annotation CSV for the requested mode."""
    annotation_dir = Path(annotation_dir)
    base = annotation_dir / f"{stem}.csv"
    reviewed = annotation_dir / f"{stem}_reviewed.csv"
    expanded = annotation_dir / f"{stem}_expanded.csv"

    mode = normalize_label_mode(mode)
    if mode == "expanded":
        for path in (expanded, reviewed):
            if path.exists():
                return path
        if allow_bootstrap_fallback and base.exists():
            return base
        raise FileNotFoundError(
            f"Expanded labels requested for '{stem}', but neither {expanded.name} nor "
            f"{reviewed.name} exists in {annotation_dir}."
        )
    if mode == "manual":
        if reviewed.exists():
            return reviewed
        if allow_bootstrap_fallback and base.exists():
            return base
        raise FileNotFoundError(
            f"Manual/reviewed labels requested for '{stem}', but {reviewed.name} does not exist "
            f"in {annotation_dir}. Set training.allow_bootstrap_fallback=true only for debug/bootstrap runs."
        )
    if mode == "auto":
        for path in (reviewed, expanded):
            if path.exists():
                return path
        if allow_bootstrap_fallback and base.exists():
            return base
        raise FileNotFoundError(
            f"Auto label resolution for '{stem}' did not find reviewed annotations in {annotation_dir}."
        )
    if mode == "bootstrap":
        for path in (base, reviewed, expanded):
            if path.exists():
                return path
        raise FileNotFoundError(f"No annotation CSV found for '{stem}' in {annotation_dir}.")
    raise ValueError(f"Unsupported label mode: {mode}")


def select_label_subset(
    df: pd.DataFrame,
    mode: str = "manual",
    allow_bootstrap_fallback: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Filter labels to the subset appropriate for the requested training mode."""
    mode = normalize_label_mode(mode)

    if "label_source" not in df.columns:
        if mode == "bootstrap":
            return df.copy(), "all(no_label_source)"
        if allow_bootstrap_fallback:
            return df.copy(), "bootstrap_fallback(no_label_source)"
        raise ValueError(
            "Requested reviewed/expanded labels, but the annotation table has no 'label_source' column."
        )

    manual_mask = df["label_source"].isin(MANUAL_SOURCES)
    expanded_mask = df["label_source"].isin(EXPANDED_SOURCES)

    if mode == "manual":
        if manual_mask.any():
            return df.loc[manual_mask].copy(), "manual_only"
        if allow_bootstrap_fallback:
            return df.copy(), "bootstrap_fallback(no_manual_labels)"
        raise ValueError(
            "Manual label mode requested, but no rows with reviewed/manual label_source are present."
        )

    if mode == "expanded":
        if expanded_mask.any():
            return df.loc[expanded_mask].copy(), "manual_plus_high_confidence"
        if manual_mask.any():
            return df.loc[manual_mask].copy(), "manual_only(no_high_confidence_yet)"
        if allow_bootstrap_fallback:
            return df.copy(), "bootstrap_fallback(no_expanded_labels)"
        raise ValueError(
            "Expanded label mode requested, but no reviewed or high-confidence pseudo-label rows are present."
        )

    if mode == "auto":
        if manual_mask.any():
            return df.loc[manual_mask].copy(), "manual_only(auto)"
        if allow_bootstrap_fallback:
            return df.copy(), "bootstrap_fallback(auto)"
        raise ValueError(
            "Auto label mode found no reviewed/manual rows and bootstrap fallback is disabled."
        )

    if mode == "bootstrap":
        return df.copy(), "all_labels"

    raise ValueError(f"Unsupported label mode: {mode}")
