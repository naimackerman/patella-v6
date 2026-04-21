"""Create a provisional non-clinician review sheet for the initial experiment."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig


def _confidence_from_margin(score: float, thresholds: list[float], default_margin: float) -> str:
    if pd.isna(score):
        return "uncertain"
    if not thresholds:
        return "uncertain"
    margin = min(abs(float(score) - float(th)) for th in thresholds)
    return "certain" if margin >= default_margin else "uncertain"


def _thresholds_for_key(calibration_section, key: str) -> list[float]:
    if not isinstance(calibration_section, dict):
        return []
    if "thresholds" in calibration_section:
        return list(calibration_section.get("thresholds", []) or [])
    nested = calibration_section.get(key, {})
    if isinstance(nested, dict):
        return list(nested.get("thresholds", []) or [])
    return []


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    annotation_dir = Path(cfg.annotation_dir)
    package_dir = annotation_dir / "packages" / "feature_grading"
    template_path = package_dir / "feature_review_template.csv"
    osp_path = annotation_dir / "osteophyte_labels.csv"
    scl_path = annotation_dir / "sclerosis_labels.csv"
    calibration_path = annotation_dir / "heuristic_calibration.json"

    if not template_path.exists():
        raise FileNotFoundError(f"Missing feature review template: {template_path}")
    if not osp_path.exists() or not scl_path.exists():
        raise FileNotFoundError("Run bootstrap_pseudo_labels.py first to generate heuristic suggestions.")

    template = pd.read_csv(template_path)
    osp = pd.read_csv(osp_path)[["image_id", "score_mf", "score_lf", "score_mt", "score_lt"]]
    scl = pd.read_csv(scl_path)[["image_id", "score_medial", "score_lateral"]]
    merged = template.merge(osp, on="image_id", how="left").merge(scl, on="image_id", how="left")

    osp_thresholds = {}
    scl_thresholds = {}
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
        osp_section = calibration.get("osteophyte", {})
        scl_section = calibration.get("sclerosis", {})
        osp_thresholds = {short: _thresholds_for_key(osp_section, site) for short, site in {
            "mf": "medial_femur",
            "lf": "lateral_femur",
            "mt": "medial_tibia",
            "lt": "lateral_tibia",
        }.items()}
        scl_thresholds = {side: _thresholds_for_key(scl_section, side) for side in ("medial", "lateral")}

    for short in ("mf", "lf", "mt", "lt"):
        final_col = f"final_osp_{short}"
        suggestion_col = f"suggestion_osp_{short}"
        confidence_col = f"confidence_{short}"
        score_col = f"score_{short}"
        merged[final_col] = merged[final_col].where(merged[final_col].notna(), merged[suggestion_col])
        merged[confidence_col] = merged.apply(
            lambda row: row[confidence_col]
            if pd.notna(row[confidence_col]) and str(row[confidence_col]).strip() != ""
            else _confidence_from_margin(row[score_col], osp_thresholds.get(short, []), default_margin=0.01),
            axis=1,
        )

    for side in ("medial", "lateral"):
        final_col = f"final_scl_{side}"
        suggestion_col = f"suggestion_scl_{side}"
        confidence_col = f"scl_confidence_{'med' if side == 'medial' else 'lat'}"
        score_col = f"score_{side}"
        merged[final_col] = merged[final_col].where(merged[final_col].notna(), merged[suggestion_col])
        merged[confidence_col] = merged.apply(
            lambda row: row[confidence_col]
            if pd.notna(row[confidence_col]) and str(row[confidence_col]).strip() != ""
            else _confidence_from_margin(row[score_col], scl_thresholds.get(side, []), default_margin=0.08),
            axis=1,
        )

    merged["notes"] = merged["notes"].where(
        merged["notes"].notna() & (merged["notes"].astype(str).str.strip() != ""),
        "auto_prefill_for_initial_experiment_non_clinician",
    )

    out_path = package_dir / "feature_review_provisional.csv"
    merged.drop(columns=["score_mf", "score_lf", "score_mt", "score_lt", "score_medial", "score_lateral"]).to_csv(out_path, index=False)

    print(f"Saved provisional review sheet: {out_path}")
    print("This file is suitable for an initial experiment only and should be clinician-reviewed later.")


if __name__ == "__main__":
    main()
