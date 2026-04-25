"""Generate publication-ready manuscript figures 2-5.

The figures are built from saved experiment artifacts so the paper images can
be regenerated without rerunning model training.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.patches import Patch, Rectangle

from src.features.feature_aggregator import FeatureAggregator
from src.features.jsw_computation import (
    compute_compartment_measurements,
    extract_contours,
)
from src.features.subchondral_roi import extract_subchondral_roi_with_boxes


KL_NAMES = ["KL0", "KL1", "KL2", "KL3", "KL4"]
OSTEOPHYTE_SITES = [
    ("medial_femur", "Medial femur"),
    ("lateral_femur", "Lateral femur"),
    ("medial_tibia", "Medial tibia"),
    ("lateral_tibia", "Lateral tibia"),
]
ABLATION_LABELS = [
    ("jsn_only", "JSN"),
    ("osp_only", "Osteophyte"),
    ("scl_only", "Sclerosis"),
    ("osp_scl", "Osteophyte +\nSclerosis"),
    ("jsn_osp", "JSN +\nOsteophyte"),
    ("full", "Full"),
]
JSN_EXAMPLES = [
    (0, "9243046R"),
    (2, "9057327R"),
    (4, "9048789L"),
]
SCLEROSIS_THRESHOLD = 0.4227197766304016
SCLEROSIS_CLASS_NAMES = ["none", "present"]
SCLEROSIS_ROI_KWARGS = {
    "depth_px": 20,
    "depth_fraction": 0.12,
    "medial_depth_fraction": 0.12,
    "lateral_depth_fraction": 0.10,
    "offset_pct": 0.10,
    "medial_offset_pct": 0.10,
    "lateral_offset_pct": 0.16,
    "medial_inner_offset_pct": 0.10,
    "medial_outer_offset_pct": 0.10,
    "lateral_inner_offset_pct": 0.10,
    "lateral_outer_offset_pct": 0.24,
    "surface_offset_fraction": 0.015,
    "surface_smoothing_window": 5,
    "output_size": 96,
}

COMPARTMENT_COLORS = {
    1: {
        "name": "Medial JSN",
        "fill": np.array([0.00, 0.62, 0.78]),
        "contour": (0, 142, 172),
        "edge": (65, 210, 230),
        "hex": "#009cbf",
    },
    2: {
        "name": "Lateral JSN",
        "fill": np.array([0.48, 0.31, 0.86]),
        "contour": (122, 86, 214),
        "edge": (174, 142, 242),
        "hex": "#7950dc",
    },
}

def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _clean_axes(ax: plt.Axes) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _image_path(project_root: Path, grade: int, image_id: str) -> Path:
    return project_root / "KneeXrayData" / "ClsKLData" / "kneeKL224" / "test" / str(grade) / f"{image_id}.png"


def _mask_path(project_root: Path, image_id: str) -> Path:
    pred_mask = project_root / "features" / "jsn" / "masks" / f"{image_id}_mask.npy"
    if pred_mask.exists():
        return pred_mask

    manual_root = project_root / "annotations" / "local" / "jsn_masks" / "test"
    for suffix in ("", " 2"):
        manual_mask = manual_root / f"{image_id}{suffix}.png"
        if manual_mask.exists():
            return manual_mask
    raise FileNotFoundError(f"No JSN mask found for {image_id}")


def _load_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path).astype(np.uint8)
    mask = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        raise ValueError(f"Could not read mask: {path}")
    return mask.astype(np.uint8)


def _mask_overlay(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    base = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32) / 255.0
    overlay = base.copy()
    for class_id, color_spec in COMPARTMENT_COLORS.items():
        overlay[mask == class_id] = color_spec["fill"]
    alpha = 0.58
    blended = base * (1.0 - alpha) + overlay * alpha
    blended[mask == 0] = base[mask == 0]
    return np.clip(blended, 0.0, 1.0)


def _draw_contours_and_measurements(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)

    all_pairs: list[tuple[int, tuple[float, float], tuple[float, float], float]] = []
    for class_id in (1, 2):
        profile, pairs = compute_compartment_measurements(mask, class_id=class_id, n_points=8)
        all_pairs.extend((class_id, p1, p2, dist) for p1, p2, dist in pairs)

        femoral, tibial = extract_contours(mask, class_id=class_id)
        if femoral is None or tibial is None:
            continue
        color_spec = COMPARTMENT_COLORS[class_id]
        for contour, color in ((femoral, color_spec["contour"]), (tibial, color_spec["edge"])):
            pts = contour.round().astype(np.int32).reshape(-1, 1, 2)
            cv2.polylines(canvas, [pts], isClosed=False, color=color, thickness=2, lineType=cv2.LINE_AA)

    for class_id, p1, p2, _dist in all_pairs:
        color = COMPARTMENT_COLORS[class_id]["contour"]
        pt1 = tuple(np.round(p1).astype(int))
        pt2 = tuple(np.round(p2).astype(int))
        cv2.line(canvas, pt1, pt2, color=color, thickness=1, lineType=cv2.LINE_AA)
        cv2.circle(canvas, pt1, 2, color=color, thickness=-1, lineType=cv2.LINE_AA)

    return canvas


def _plot_profile(ax: plt.Axes, mask: np.ndarray, show_ylabel: bool = False) -> None:
    plotted = False
    for class_id in (1, 2):
        profile, _ = compute_compartment_measurements(mask, class_id=class_id, n_points=16)
        x = np.arange(1, len(profile) + 1)
        if np.any(profile > 0):
            ax.plot(
                x,
                profile,
                marker="o",
                linewidth=1.8,
                markersize=3.2,
                color=COMPARTMENT_COLORS[class_id]["hex"],
                label=COMPARTMENT_COLORS[class_id]["name"].replace(" JSN", ""),
            )
            plotted = True
    ax.set_xlabel("Sample point", fontsize=8)
    if show_ylabel:
        ax.set_ylabel("JSW (px)", fontsize=8)
    else:
        ax.set_ylabel("")
    ax.tick_params(axis="both", labelsize=7)
    ax.grid(alpha=0.25, linewidth=0.6)


def figure_jsn(project_root: Path, out_dir: Path) -> None:
    fig, axes = plt.subplots(
        4,
        len(JSN_EXAMPLES),
        figsize=(9.0, 8.2),
        gridspec_kw={"height_ratios": [1.0, 1.0, 1.0, 0.78], "hspace": 0.12, "wspace": 0.25},
    )
    row_labels = ["Radiograph", "Predicted JSN mask", "Contours + mJSW", "JSW profile"]

    for col, (grade, image_id) in enumerate(JSN_EXAMPLES):
        image = cv2.imread(str(_image_path(project_root, grade, image_id)), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read image for {image_id}")
        mask = _load_mask(_mask_path(project_root, image_id))

        axes[0, col].imshow(image, cmap="gray", vmin=0, vmax=255)
        axes[1, col].imshow(_mask_overlay(image, mask))
        axes[2, col].imshow(_draw_contours_and_measurements(image, mask))
        _plot_profile(axes[3, col], mask, show_ylabel=col == 0)

        axes[0, col].set_title(f"KL {grade}: {image_id}", fontsize=10, fontweight="bold", pad=8)
        for row in range(3):
            _clean_axes(axes[row, col])

    for row, label in enumerate(row_labels):
        axes[row, 0].set_ylabel(label, fontsize=9, fontweight="bold")

    legend = [
        Patch(facecolor=COMPARTMENT_COLORS[1]["hex"], label=COMPARTMENT_COLORS[1]["name"]),
        Patch(facecolor=COMPARTMENT_COLORS[2]["hex"], label=COMPARTMENT_COLORS[2]["name"]),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, 0.005))
    profile_handles = [
        plt.Line2D([0], [0], color=COMPARTMENT_COLORS[1]["hex"], marker="o", linewidth=1.8, markersize=3.2, label="Medial"),
        plt.Line2D([0], [0], color=COMPARTMENT_COLORS[2]["hex"], marker="o", linewidth=1.8, markersize=3.2, label="Lateral"),
    ]
    axes[3, 1].legend(handles=profile_handles, frameon=False, fontsize=7, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.18))
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.08)
    _save_figure(fig, out_dir, "fig2_jsn_vis")


def figure_osteophyte(project_root: Path, out_dir: Path) -> None:
    payload = _load_json(project_root / "results" / "osteophyte_main_manual" / "osteophyte_evaluation.json")
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 7.4), constrained_layout=True)
    for ax, (key, title) in zip(axes.flat, OSTEOPHYTE_SITES):
        site = payload[key]["test"]
        cm = np.asarray(site["confusion_matrix"], dtype=int)
        row_totals = cm.sum(axis=1, keepdims=True)
        row_pct = np.divide(cm, row_totals, out=np.zeros_like(cm, dtype=float), where=row_totals > 0) * 100.0
        annot = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annot[i, j] = f"{row_pct[i, j]:.0f}%\n({cm[i, j]})"
        sns.heatmap(
            row_pct,
            annot=annot,
            fmt="",
            cmap="Blues",
            cbar=False,
            vmin=0.0,
            vmax=100.0,
            square=True,
            linewidths=0.5,
            linecolor="white",
            xticklabels=[0, 1, 2, 3],
            yticklabels=[0, 1, 2, 3],
            annot_kws={"fontsize": 7},
            ax=ax,
        )
        ax.set_title(f"{title}\n$\\kappa$={site['kappa']:.4f}, AUC={site['auc_macro']:.4f}", fontsize=10)
        ax.set_xlabel("Predicted OARSI grade", fontsize=9)
        ax.set_ylabel("True OARSI grade", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
    _save_figure(fig, out_dir, "fig3_osteophyte_cm")


def _local_artifact_path(project_root: Path, value: object) -> Path:
    return Path(str(value).replace("/workspace/patella-v6", str(project_root)))


def _test_image_lookup(project_root: Path) -> dict[str, tuple[int, Path]]:
    test_root = project_root / "KneeXrayData" / "ClsKLData" / "kneeKL224" / "test"
    lookup: dict[str, tuple[int, Path]] = {}
    for grade_dir in sorted(test_root.glob("*")):
        if not grade_dir.is_dir() or not grade_dir.name.isdigit():
            continue
        grade = int(grade_dir.name)
        for image_path in grade_dir.glob("*.png"):
            lookup[image_path.stem] = (grade, image_path)
    return lookup


def _load_sclerosis_cases(project_root: Path) -> list[dict]:
    import torch
    from omegaconf import OmegaConf

    from src.models.sclerosis_classifier import SclerosisClassifier
    from src.utils.checkpoints import extract_model_state_dict, load_checkpoint
    from src.utils.feature_scaling import load_standardizer, transform_with_standardizer
    from src.utils.sclerosis_labels import map_sclerosis_grades

    data = np.load(project_root / "features" / "sclerosis_manual_teacher" / "test_sclerosis_data.npz", allow_pickle=True)
    manual_idx = np.flatnonzero(data["label_sources"].astype(str) == "manual_review")
    if manual_idx.size == 0:
        raise ValueError("No manual-review sclerosis cases found.")

    standardizer = load_standardizer(project_root / "features" / "sclerosis_manual_teacher" / "texture_standardizer.npz")
    if standardizer is None:
        raise FileNotFoundError("Missing sclerosis texture standardizer.")
    mean, scale = standardizer
    texture_features = transform_with_standardizer(data["texture_features"][manual_idx], mean, scale).astype(np.float32)
    side_ids = data["side_ids"][manual_idx].astype(np.int64)
    labels = map_sclerosis_grades(data["grades"][manual_idx], "binary_present")

    model_cfg = OmegaConf.load(project_root / "configs" / "model" / "sclerosis_hybrid.yaml")
    model_cfg = OmegaConf.merge(
        model_cfg,
        {
            "input_mode": "texture_only",
            "pretrained": False,
            "num_classes": 2,
            "class_names": SCLEROSIS_CLASS_NAMES,
        },
    )
    model = SclerosisClassifier(model_cfg)
    checkpoint = load_checkpoint(
        project_root
        / "checkpoints"
        / "sclerosis_binary_texture_only"
        / "sclerosis"
        / "scl-auc-epoch=044-val_auc_macro=0.6730.ckpt",
        map_location="cpu",
    )
    model.load_state_dict(extract_model_state_dict(checkpoint))
    model.eval()

    with torch.no_grad():
        logits = model(
            roi_image=torch.zeros((manual_idx.size, 1, 96, 96), dtype=torch.float32),
            texture_features=torch.from_numpy(texture_features),
            side_ids=torch.from_numpy(side_ids),
        )
        probabilities = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
    predictions = (probabilities >= SCLEROSIS_THRESHOLD).astype(np.int64)

    image_lookup = _test_image_lookup(project_root)
    grouped: dict[str, dict] = {}
    for row, idx in enumerate(manual_idx):
        image_id = str(data["image_ids"][idx])
        base_id, side_name = image_id.rsplit("_", 1)
        if base_id not in image_lookup:
            continue
        case = grouped.setdefault(
            base_id,
            {
                "base_id": base_id,
                "grade": image_lookup[base_id][0],
                "image_path": image_lookup[base_id][1],
                "mask_path": _mask_path(project_root, base_id),
                "sides": {},
            },
        )
        case["sides"][side_name] = {
            "probability": float(probabilities[row]),
            "prediction": int(predictions[row]),
            "true_label": int(labels[row]),
            "patch_path": _local_artifact_path(project_root, data["roi_paths"][idx]),
        }

    return [
        case
        for case in grouped.values()
        if {"medial", "lateral"}.issubset(case["sides"])
        and Path(case["image_path"]).exists()
        and Path(case["mask_path"]).exists()
        and Path(case["sides"]["medial"]["patch_path"]).exists()
        and Path(case["sides"]["lateral"]["patch_path"]).exists()
    ]


def _select_sclerosis_examples(cases: list[dict]) -> list[tuple[str, dict]]:
    def mean_probability(case: dict) -> float:
        return float(np.mean([case["sides"]["medial"]["probability"], case["sides"]["lateral"]["probability"]]))

    def pred_sum(case: dict) -> int:
        return int(case["sides"]["medial"]["prediction"] + case["sides"]["lateral"]["prediction"])

    selected: list[tuple[str, dict]] = []
    none_cases = [case for case in cases if pred_sum(case) == 0]
    mixed_cases = [case for case in cases if pred_sum(case) == 1]
    present_cases = [case for case in cases if pred_sum(case) == 2]

    if none_cases:
        selected.append(("Low-probability none", sorted(none_cases, key=mean_probability)[0]))
    if mixed_cases:
        selected.append(
            (
                "Asymmetric output",
                sorted(
                    mixed_cases,
                    key=lambda case: abs(
                        case["sides"]["medial"]["probability"] - case["sides"]["lateral"]["probability"]
                    ),
                    reverse=True,
                )[0],
            )
        )
    if present_cases:
        selected.append(("High-probability present", sorted(present_cases, key=mean_probability, reverse=True)[0]))

    used = {case["base_id"] for _, case in selected}
    for case in sorted(cases, key=lambda item: abs(mean_probability(item) - SCLEROSIS_THRESHOLD)):
        if len(selected) >= 3:
            break
        if case["base_id"] not in used:
            selected.append(("Near threshold", case))
            used.add(case["base_id"])
    if len(selected) < 3:
        raise ValueError("Could not find three representative sclerosis examples.")
    return selected[:3]


def _draw_sclerosis_roi_overlay(image: np.ndarray, mask: np.ndarray, image_id: str) -> np.ndarray:
    canvas = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    _medial_roi, _lateral_roi, medial_box, lateral_box = extract_subchondral_roi_with_boxes(
        mask=mask,
        image=image,
        is_left=image_id.endswith("L"),
        **SCLEROSIS_ROI_KWARGS,
    )
    for label, box, color_spec in (
        ("M", medial_box, COMPARTMENT_COLORS[1]),
        ("L", lateral_box, COMPARTMENT_COLORS[2]),
    ):
        if box is None:
            continue
        x1, y1, x2, y2 = box
        color = color_spec["edge"]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), color=color, thickness=2)
        cv2.putText(canvas, label, (x1 + 2, max(y1 - 5, 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.42, color, 1, cv2.LINE_AA)
    return canvas


def _read_roi_patch(path: Path) -> np.ndarray:
    patch = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if patch is None:
        raise ValueError(f"Could not read ROI patch: {path}")
    return patch


def figure_sclerosis_roi(project_root: Path, out_dir: Path) -> None:
    cases = _select_sclerosis_examples(_load_sclerosis_cases(project_root))
    fig = plt.figure(figsize=(10.2, 6.2))
    gs = fig.add_gridspec(
        3,
        4,
        width_ratios=[1.15, 0.72, 0.72, 1.05],
        left=0.035,
        right=0.99,
        top=0.92,
        bottom=0.09,
        wspace=0.16,
        hspace=0.20,
    )
    column_titles = ["Radiograph + ROI", "Medial ROI", "Lateral ROI", "Model output"]

    for row, (case_role, case) in enumerate(cases):
        image = cv2.imread(str(case["image_path"]), cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise ValueError(f"Could not read radiograph: {case['image_path']}")
        mask = _load_mask(case["mask_path"])

        axes = [fig.add_subplot(gs[row, col]) for col in range(4)]
        axes[0].imshow(_draw_sclerosis_roi_overlay(image, mask, case["base_id"]))
        axes[1].imshow(_read_roi_patch(case["sides"]["medial"]["patch_path"]), cmap="gray", vmin=0, vmax=255)
        axes[2].imshow(_read_roi_patch(case["sides"]["lateral"]["patch_path"]), cmap="gray", vmin=0, vmax=255)

        for col, ax in enumerate(axes[:3]):
            _clean_axes(ax)
            if row == 0:
                ax.set_title(column_titles[col], fontsize=9, fontweight="bold", pad=6)

        axes[0].text(
            0.02,
            0.97,
            f"{case_role}\nKL {case['grade']}: {case['base_id']}",
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=7.6,
            fontweight="bold",
            color="white",
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "black", "edgecolor": "none", "alpha": 0.58},
        )
        axes[3].axis("off")
        if row == 0:
            axes[3].set_title(column_titles[3], fontsize=9, fontweight="bold", pad=6)

        medial = case["sides"]["medial"]
        lateral = case["sides"]["lateral"]
        output_lines = [
            f"Medial: {SCLEROSIS_CLASS_NAMES[medial['prediction']]}",
            f"p(present)={medial['probability']:.4f}",
            "",
            f"Lateral: {SCLEROSIS_CLASS_NAMES[lateral['prediction']]}",
            f"p(present)={lateral['probability']:.4f}",
            "",
            f"Threshold={SCLEROSIS_THRESHOLD:.4f}",
        ]
        axes[3].text(
            0.02,
            0.52,
            "\n".join(output_lines),
            transform=axes[3].transAxes,
            ha="left",
            va="center",
            fontsize=8,
            linespacing=1.35,
            bbox={"boxstyle": "round,pad=0.32", "facecolor": "#f8f9fa", "edgecolor": "#ced4da", "linewidth": 0.7},
        )

    legend = [
        Patch(facecolor=COMPARTMENT_COLORS[1]["hex"], label="Medial tibial ROI"),
        Patch(facecolor=COMPARTMENT_COLORS[2]["hex"], label="Lateral tibial ROI"),
    ]
    fig.legend(handles=legend, loc="lower center", ncol=2, frameon=False, fontsize=8, bbox_to_anchor=(0.5, -0.02))
    _save_figure(fig, out_dir, "fig4_perclass_f1")


def figure_ablation_shap(project_root: Path, out_dir: Path) -> None:
    ablation = np.load(project_root / "results" / "ablation" / "ablation_results.npz")

    labels = [label for _, label in ABLATION_LABELS]
    qwk = [float(ablation[key][0]) for key, _ in ABLATION_LABELS]
    colors = ["#2f9f67", "#5c7cfa", "#f06595", "#9775fa", "#20c997", "#343a40"]

    fig = plt.figure(figsize=(11.0, 5.6), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.18])
    ax_bar = fig.add_subplot(gs[0, 0])
    ax_shap = fig.add_subplot(gs[0, 1])

    bars = ax_bar.bar(np.arange(len(labels)), qwk, color=colors, width=0.72)
    for bar, value in zip(bars, qwk):
        ax_bar.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.4f}", ha="center", va="bottom", fontsize=8)
    ax_bar.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right", fontsize=8)
    ax_bar.set_ylabel("Test QWK", fontsize=10)
    ax_bar.set_ylim(0, 0.72)
    ax_bar.grid(axis="y", alpha=0.25, linewidth=0.7)
    ax_bar.set_title("A. Feature-family ablation", loc="left", fontsize=11, fontweight="bold")
    ax_bar.spines["top"].set_visible(False)
    ax_bar.spines["right"].set_visible(False)

    _plot_predicted_class_shap_beeswarm(ax_shap, project_root)
    ax_shap.set_title("B. Global SHAP feature attribution", loc="left", fontsize=11, fontweight="bold")
    _save_figure(fig, out_dir, "fig5_ablation_shap")


def _short_feature_name(name: str) -> str:
    replacements = {
        "mJSW_": "mJSW ",
        "jsw_profile_": "JSW profile ",
        "jsn_rate_": "JSN rate ",
        "jsw_": "JSW ",
        "osp_grade_": "OSP grade ",
        "osp_": "OSP ",
        "scl_grade_": "SCL grade ",
        "scl_intensity_": "SCL intensity ",
        "scl_fractal_dim_": "SCL fractal ",
        "scl_glcm_": "SCL GLCM ",
        "scl_lbp_entropy_": "SCL LBP entropy ",
    }
    for old, new in replacements.items():
        if name.startswith(old):
            name = new + name[len(old) :]
            break
    return name.replace("_", " ")


def _plot_predicted_class_shap_beeswarm(ax: plt.Axes, project_root: Path) -> None:
    import shap
    import xgboost as xgb

    train = np.load(project_root / "features" / "aggregated" / "train_features.npz")
    test = np.load(project_root / "features" / "aggregated" / "test_features.npz")

    aggregator = FeatureAggregator()
    aggregator.fit_normalizer(train["features"])
    X_test = aggregator.normalize(test["features"])
    feature_names = aggregator.get_feature_names()

    max_samples = min(300, X_test.shape[0])
    sample_idx = np.linspace(0, X_test.shape[0] - 1, max_samples).round().astype(int)
    X_sample = X_test[sample_idx]

    model = xgb.Booster()
    model.load_model(str(project_root / "checkpoints" / "kl_xgboost.ubj"))
    preds = np.argmax(model.predict(xgb.DMatrix(X_sample)), axis=1).astype(int)

    explainer = shap.TreeExplainer(model)
    shap_values = np.asarray(explainer.shap_values(X_sample))
    if shap_values.ndim != 3:
        raise ValueError(f"Expected multiclass SHAP array with 3 dims, got {shap_values.shape}")
    shap_for_pred = shap_values[np.arange(X_sample.shape[0]), :, preds]

    top_n = 12
    order = np.argsort(np.abs(shap_for_pred).mean(axis=0))[-top_n:][::-1]
    y_positions = np.arange(top_n)
    rng = np.random.default_rng(42)

    scatter = None
    for y, feature_idx in zip(y_positions, order):
        values = shap_for_pred[:, feature_idx]
        feature_values = X_sample[:, feature_idx]
        lo, hi = np.percentile(feature_values, [2, 98])
        if hi <= lo:
            colors = np.zeros_like(feature_values)
        else:
            colors = np.clip((feature_values - lo) / (hi - lo), 0, 1)
        jitter = rng.uniform(-0.28, 0.28, size=values.shape[0])
        scatter = ax.scatter(
            values,
            np.full_like(values, y, dtype=float) + jitter,
            c=colors,
            cmap="coolwarm",
            s=7,
            alpha=0.62,
            edgecolors="none",
            rasterized=True,
        )

    ax.axvline(0, color="#495057", linewidth=0.8)
    ax.set_yticks(y_positions, [_short_feature_name(feature_names[idx]) for idx in order])
    ax.invert_yaxis()
    ax.set_xlabel("SHAP contribution to predicted KL class", fontsize=9)
    ax.tick_params(axis="both", labelsize=8)
    ax.grid(axis="x", alpha=0.20, linewidth=0.6)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if scatter is not None:
        cbar = plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.02)
        cbar.set_label("Feature value", fontsize=8)
        cbar.set_ticks([0, 1], labels=["Low", "High"])
        cbar.ax.tick_params(labelsize=7)


def _save_figure(fig: plt.Figure, out_dir: Path, stem: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(out_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out-dir", type=Path, default=Path("paper_figures"))
    parser.add_argument("--figure", choices=["all", "fig2", "fig3", "fig4", "fig5"], default="all")
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    out_dir = args.out_dir
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir

    figure_map = {
        "fig2": figure_jsn,
        "fig3": figure_osteophyte,
        "fig4": figure_sclerosis_roi,
        "fig5": figure_ablation_shap,
    }
    if args.figure == "all":
        for figure_name in ("fig2", "fig3", "fig5", "fig4"):
            subprocess.run(
                [
                    sys.executable,
                    str(Path(__file__).resolve()),
                    "--project-root",
                    str(project_root),
                    "--out-dir",
                    str(out_dir),
                    "--figure",
                    figure_name,
                ],
                check=True,
            )
    else:
        figure_map[args.figure](project_root, out_dir)
    print(f"Generated manuscript figures in {out_dir}")


if __name__ == "__main__":
    main()
