"""Pipeline readiness audit aligned with the KOA-TriFQ research workflow."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from src.models.kl_xgboost import KLXGBoostClassifier


def _artifact(
    label: str,
    *,
    path: str | None = None,
    glob: str | None = None,
    required: bool = True,
) -> dict:
    return {
        "label": label,
        "path": path,
        "glob": glob,
        "required": required,
    }


def _resolve_artifact(project_root: Path, spec: dict) -> dict:
    relative_path = spec.get("path")
    glob_pattern = spec.get("glob")

    matches: list[str] = []
    present = False
    if relative_path is not None:
        artifact_path = project_root / relative_path
        present = artifact_path.exists()
        display_path = relative_path
    elif glob_pattern is not None:
        matches = sorted(str(path.relative_to(project_root)) for path in project_root.glob(glob_pattern))
        present = len(matches) > 0
        display_path = glob_pattern
    else:
        raise ValueError("Artifact spec must define either 'path' or 'glob'.")

    return {
        "label": spec["label"],
        "path": display_path,
        "present": present,
        "required": bool(spec.get("required", True)),
        "matches": matches[:5],
        "match_count": len(matches),
    }


def _build_stage(project_root: Path, name: str, title: str, artifacts: Iterable[dict]) -> dict:
    resolved = [_resolve_artifact(project_root, spec) for spec in artifacts]
    required = [artifact for artifact in resolved if artifact["required"]]
    required_present = sum(int(artifact["present"]) for artifact in required)
    required_total = len(required)

    if required_total == 0:
        status = "ready"
    elif required_present == required_total:
        status = "ready"
    elif required_present == 0:
        status = "missing"
    else:
        status = "partial"

    return {
        "name": name,
        "title": title,
        "status": status,
        "required_present": required_present,
        "required_total": required_total,
        "artifacts": resolved,
    }


def _stage_status(stage_groups: dict[str, list[dict]], name: str) -> str:
    for stages in stage_groups.values():
        for stage in stages:
            if stage["name"] == name:
                return str(stage["status"])
    raise KeyError(f"Unknown stage: {name}")


def _cmd(command: str) -> str:
    return f"PYTHONPATH=. python3 {command}"


def _collect_recommended_commands(stage_groups: dict[str, list[dict]]) -> list[str]:
    commands: list[str] = []

    if _stage_status(stage_groups, "jsn_stage") != "ready":
        commands.extend([
            _cmd("scripts/train_jsn_segmenter.py training.label_mode=manual +model=unetpp"),
            _cmd("scripts/extract_jsn_features.py training.label_mode=manual +model=unetpp"),
            _cmd("scripts/evaluate_jsn_segmenter.py +model=unetpp"),
        ])
    if _stage_status(stage_groups, "roi_pipeline") != "ready":
        commands.append(
            _cmd(
                "scripts/extract_roi_with_fullimage_clahe.py "
                "--input-dir KneeXrayData/ClsKLData/kneeKL224 "
                "--output-dir features/rois_osteophyte_clahe_full "
                "--scan-all-images "
                "--reviewed-jsn-mask-dir annotations/jsn_masks "
                "--predicted-jsn-mask-dir features/jsn/masks "
                "--clahe-clip 3.0 "
                "--clahe-tile 8"
            )
        )
    if _stage_status(stage_groups, "osteophyte_stage") != "ready":
        commands.extend([
            _cmd(
                "scripts/train_osteophyte_grader.py "
                "training.label_mode=manual "
                "preprocessing.augmentation.horizontal_flip_p=0.0 "
                "preprocessing.clahe=null "
                "preprocessing.histogram_clip=null "
                "training.scheduler=reduce_on_plateau "
                "training.scheduler_params.mode=max "
                "training.scheduler_params.factor=0.5 "
                "training.scheduler_params.patience=10 "
                "training.layer_wise_lr.enabled=true "
                "training.layer_wise_lr.backbone_ratio=0.1 "
                "training.osteophyte_class_balance.enabled=true "
                "training.osteophyte_sampling.strategy=mean_class_balance "
                "training.osteophyte_refinement.sites='[lateral_femur,medial_tibia]' "
                "training.osteophyte_warm_start_checkpoint=checkpoints/clahe_fullimage_ordinal/hybrid-epoch=078-val_kappa_mean=0.8264.ckpt "
                "training.osteophyte_force_retrain_multitask=true "
                "osteophyte_roi_dir=features/rois_osteophyte_clahe_full"
            ),
            _cmd(
                "scripts/extract_osteophyte_features.py "
                "training.label_mode=manual "
                "preprocessing.clahe=null "
                "preprocessing.histogram_clip=null "
                "osteophyte_roi_dir=features/rois_osteophyte_clahe_full"
            ),
            _cmd(
                "scripts/evaluate_osteophyte_grader.py "
                "training.label_mode=manual "
                "preprocessing.clahe=null "
                "preprocessing.histogram_clip=null "
                "osteophyte_roi_dir=features/rois_osteophyte_clahe_full "
                "result_dir=results/osteophyte_main_manual"
            ),
        ])
    if _stage_status(stage_groups, "sclerosis_stage") != "ready":
        commands.extend([
            _cmd("scripts/extract_sclerosis_features.py training.label_mode=manual +model=sclerosis_hybrid"),
            _cmd("scripts/train_sclerosis.py training.label_mode=manual +model=sclerosis_hybrid"),
            _cmd("scripts/extract_sclerosis_features.py training.label_mode=manual +model=sclerosis_hybrid"),
            _cmd("scripts/evaluate_sclerosis.py +model=sclerosis_hybrid"),
        ])
    if _stage_status(stage_groups, "aggregation_stage") != "ready":
        commands.append(_cmd("scripts/extract_all_features.py"))
    if _stage_status(stage_groups, "kl_models") != "ready":
        commands.extend([
            _cmd("scripts/train_kl_xgboost.py +model=xgboost"),
            _cmd("scripts/train_kl_hybrid.py +model=convnext_hybrid"),
            _cmd("scripts/run_kl_feature_baselines.py"),
            _cmd("scripts/evaluate_pipeline.py +model=xgboost"),
        ])

    deduped: list[str] = []
    seen: set[str] = set()
    for command in commands:
        if command not in seen:
            seen.add(command)
            deduped.append(command)
    return deduped


def assess_pipeline_readiness(project_root: str | Path) -> dict:
    """Audit framework implementation and stage artifacts for KOA-TriFQ."""
    root = Path(project_root).resolve()
    xgb_model_path = KLXGBoostClassifier.resolve_model_path(root / "checkpoints" / "kl_xgboost")
    xgb_relative = str(xgb_model_path.relative_to(root))

    stage_groups = {
        "framework_modules": [
            _build_stage(
                root,
                "core_framework",
                "Core Quantification Modules",
                [
                    _artifact("ROI detector", path="src/models/roi_detector.py"),
                    _artifact("JSN segmenter", path="src/models/jsn_segmenter.py"),
                    _artifact("Osteophyte grader", path="src/models/osteophyte_grader.py"),
                    _artifact("Sclerosis classifier", path="src/models/sclerosis_classifier.py"),
                    _artifact("Feature aggregator", path="src/features/feature_aggregator.py"),
                ],
            ),
            _build_stage(
                root,
                "xai_and_deployment",
                "XAI and Deployment Interfaces",
                [
                    _artifact("Overlay renderer", path="src/xai/overlay_renderer.py"),
                    _artifact("Clinical report generator", path="src/xai/report_generator.py"),
                    _artifact("Single-image inference", path="app/inference.py"),
                    _artifact("Gradio app", path="app/gradio_app.py"),
                    _artifact("FastAPI app", path="app/api.py"),
                    _artifact("PDF export", path="scripts/export_pdf_report.py"),
                ],
            ),
        ],
        "study_pipeline": [
            _build_stage(
                root,
                "bootstrap_artifacts",
                "Bootstrap Annotation Seed",
                [
                    _artifact("Bootstrap osteophyte labels", path="annotations/osteophyte_labels.csv"),
                    _artifact("Bootstrap sclerosis labels", path="annotations/sclerosis_labels.csv"),
                    _artifact("Feature grading manifest", path="annotations/manifests/feature_grading_manifest.csv"),
                    _artifact("JSN contour manifest", path="annotations/manifests/jsn_contour_manifest.csv"),
                ],
            ),
            _build_stage(
                root,
                "reviewed_annotations",
                "Reviewed Manual Annotation Imports",
                [
                    _artifact("Reviewed osteophyte labels", path="annotations/osteophyte_labels_reviewed.csv"),
                    _artifact("Reviewed sclerosis labels", path="annotations/sclerosis_labels_reviewed.csv"),
                    _artifact("Reviewed JSN masks (train)", glob="annotations/jsn_masks/train/*.png"),
                    _artifact("Reviewed JSN masks (val)", glob="annotations/jsn_masks/val/*.png"),
                    _artifact("Reviewed JSN masks (test)", glob="annotations/jsn_masks/test/*.png"),
                ],
            ),
            _build_stage(
                root,
                "roi_pipeline",
                "Stage 1 Osteophyte ROI Preparation",
                [
                    _artifact("CLAHE osteophyte ROI patches (train)", glob="features/rois_osteophyte_clahe_full/train/*.png"),
                    _artifact("CLAHE osteophyte ROI patches (val)", glob="features/rois_osteophyte_clahe_full/val/*.png"),
                    _artifact("CLAHE osteophyte ROI patches (test)", glob="features/rois_osteophyte_clahe_full/test/*.png"),
                    _artifact("CLAHE extractor script", path="scripts/extract_roi_with_fullimage_clahe.py", required=False),
                ],
            ),
            _build_stage(
                root,
                "jsn_stage",
                "Stage 2A JSN Quantification",
                [
                    _artifact("JSN checkpoints", glob="checkpoints/jsn_segmenter/*.ckpt"),
                    _artifact("Train JSN features", path="features/jsn/train_jsn_features.npz"),
                    _artifact("Val JSN features", path="features/jsn/val_jsn_features.npz"),
                    _artifact("Test JSN features", path="features/jsn/test_jsn_features.npz"),
                    _artifact("Predicted JSN masks", glob="features/jsn/masks/*_mask.npy"),
                ],
            ),
            _build_stage(
                root,
                "osteophyte_stage",
                "Stage 2B Osteophyte Quantification",
                [
                    _artifact("Main osteophyte trainer", path="scripts/train_osteophyte_grader.py"),
                    _artifact("Osteophyte checkpoints", glob="checkpoints/osteophyte/*.ckpt"),
                    _artifact("Train osteophyte features", path="features/osteophyte/train_osteophyte_features.npz"),
                    _artifact("Val osteophyte features", path="features/osteophyte/val_osteophyte_features.npz"),
                    _artifact("Test osteophyte features", path="features/osteophyte/test_osteophyte_features.npz"),
                ],
            ),
            _build_stage(
                root,
                "sclerosis_stage",
                "Stage 2C Sclerosis Quantification",
                [
                    _artifact("Train sclerosis data", path="features/sclerosis/train_sclerosis_data.npz"),
                    _artifact("Val sclerosis data", path="features/sclerosis/val_sclerosis_data.npz"),
                    _artifact("Train sclerosis features", path="features/sclerosis/train_sclerosis_features.npz"),
                    _artifact("Val sclerosis features", path="features/sclerosis/val_sclerosis_features.npz"),
                    _artifact("Sclerosis checkpoints", glob="checkpoints/sclerosis/*.ckpt"),
                ],
            ),
            _build_stage(
                root,
                "aggregation_stage",
                "Stage 3 Feature Aggregation",
                [
                    _artifact("Aggregated train features", path="features/aggregated/train_features.npz"),
                    _artifact("Aggregated val features", path="features/aggregated/val_features.npz"),
                    _artifact("Aggregated test features", path="features/aggregated/test_features.npz"),
                    _artifact("Feature normalizer stats", path="features/aggregated/normalizer_stats.npz"),
                ],
            ),
            _build_stage(
                root,
                "kl_models",
                "Stage 4 KL Classification Paths",
                [
                    _artifact("Transparent XGBoost model", path=xgb_relative),
                    _artifact("Hybrid KL checkpoints", glob="checkpoints/kl_hybrid/*.ckpt"),
                ],
            ),
        ],
    }

    blockers: list[str] = []
    if _stage_status(stage_groups, "roi_pipeline") != "ready":
        blockers.append(
            "Osteophyte ROI crops under `features/rois_osteophyte_clahe_full/` are not fully prepared, "
            "so the JSN-guided HOW_TO_REPRODUCE osteophyte stage cannot run from the documented path."
        )
    if _stage_status(stage_groups, "aggregation_stage") != "ready":
        blockers.append(
            "Aggregated feature matrices under `features/aggregated/` are missing, so KL training, "
            "ablation, and end-to-end pipeline evaluation are not stage-ready."
        )
    if _stage_status(stage_groups, "kl_models") != "ready":
        blockers.append(
            "Both KL classification paths are not fully reproducible from current artifacts. "
            "At least one trained KL model path or its upstream aggregated features are incomplete."
        )

    framework_ready = sum(stage["status"] == "ready" for stage in stage_groups["framework_modules"])
    pipeline_ready = sum(stage["status"] == "ready" for stage in stage_groups["study_pipeline"])
    pipeline_partial = sum(stage["status"] == "partial" for stage in stage_groups["study_pipeline"])
    pipeline_missing = sum(stage["status"] == "missing" for stage in stage_groups["study_pipeline"])

    report = {
        "project_root": str(root),
        "summary": {
            "framework_ready": framework_ready,
            "framework_total": len(stage_groups["framework_modules"]),
            "pipeline_ready": pipeline_ready,
            "pipeline_partial": pipeline_partial,
            "pipeline_missing": pipeline_missing,
            "pipeline_total": len(stage_groups["study_pipeline"]),
        },
        "framework_modules": stage_groups["framework_modules"],
        "study_pipeline": stage_groups["study_pipeline"],
        "blockers": blockers,
        "recommended_commands": _collect_recommended_commands(stage_groups),
    }
    return report


def format_readiness_report(report: dict) -> str:
    """Render a human-readable readiness summary."""
    summary = report["summary"]
    lines = [
        "KOA-TriFQ Pipeline Readiness",
        f"Project root: {report['project_root']}",
        "",
        (
            "Framework modules: "
            f"{summary['framework_ready']}/{summary['framework_total']} ready"
        ),
        (
            "Study pipeline: "
            f"{summary['pipeline_ready']} ready, "
            f"{summary['pipeline_partial']} partial, "
            f"{summary['pipeline_missing']} missing "
            f"(total {summary['pipeline_total']})"
        ),
        "",
        "Framework Modules:",
    ]

    for stage in report["framework_modules"]:
        lines.append(
            f"- {stage['title']}: {stage['status']} "
            f"({stage['required_present']}/{stage['required_total']} required artifacts)"
        )

    lines.append("")
    lines.append("Study Pipeline:")
    for stage in report["study_pipeline"]:
        missing = [
            artifact["path"]
            for artifact in stage["artifacts"]
            if artifact["required"] and not artifact["present"]
        ]
        line = (
            f"- {stage['title']}: {stage['status']} "
            f"({stage['required_present']}/{stage['required_total']} required artifacts)"
        )
        if missing:
            line += f" | missing: {', '.join(missing)}"
        lines.append(line)

    if report["blockers"]:
        lines.append("")
        lines.append("Key Blockers:")
        for blocker in report["blockers"]:
            lines.append(f"- {blocker}")

    if report["recommended_commands"]:
        lines.append("")
        lines.append("Recommended Next Commands:")
        for command in report["recommended_commands"]:
            lines.append(f"- {command}")

    return "\n".join(lines)
