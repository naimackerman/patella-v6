"""Multi-layer XAI overlay renderer for annotated X-ray visualization."""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.features.jsw_computation import compute_measurement_pairs_from_contours


SEVERITY_COLORS = {0: None, 1: "lime", 2: "yellow", 3: "red"}
SEVERITY_LABELS = {0: "None", 1: "Small", 2: "Moderate", 3: "Large"}


def generate_xai_overlay(
    image: np.ndarray,
    jsn_results: Dict,
    osp_results: Dict,
    scl_results: Dict,
    kl_pred: Dict,
    show_jsn: bool = True,
    show_osteophytes: bool = True,
    show_sclerosis: bool = True,
    figsize: Tuple[int, int] = (8, 8),
) -> plt.Figure:
    """Generate multi-layer annotated X-ray overlay.

    Layers:
    1. Blue distance lines for JSN measurements
    2. Severity-colored bounding boxes at 4 osteophyte ROI sites
    3. Yellow-red semi-transparent heatmap for sclerosis

    Args:
        image: (H, W) grayscale X-ray image.
        jsn_results: Dict with keys: measurement_pairs, mJSW_medial, mJSW_lateral, etc.
        osp_results: Dict with keys: detections (list of (site, grade, bbox)).
        scl_results: Dict with keys: heatmap, roi_extent.
        kl_pred: Dict with keys: grade, confidence, probabilities.

    Returns:
        matplotlib Figure with the annotated overlay.
    """
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.imshow(image, cmap="gray")

    # Layer 1: JSN distance lines (blue)
    if show_jsn and "measurement_pairs" in jsn_results:
        for p1, p2, dist in jsn_results["measurement_pairs"]:
            ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "b-", linewidth=1.5, alpha=0.8)
            midx = (p1[0] + p2[0]) / 2
            midy = (p1[1] + p2[1]) / 2
            ax.annotate(
                f"{dist:.1f}px", (midx, midy),
                color="cyan", fontsize=7, ha="center",
                bbox=dict(boxstyle="round,pad=0.1", fc="black", alpha=0.5),
            )

    # Layer 2: Osteophyte severity boxes
    if show_osteophytes and "detections" in osp_results:
        for heatmap_item in osp_results.get("heatmaps", []):
            heatmap = heatmap_item.get("heatmap")
            bbox = heatmap_item.get("bbox")
            if heatmap is None or bbox is None:
                continue
            x, y, w, h = bbox
            ax.imshow(
                heatmap,
                cmap="autumn",
                alpha=0.20,
                extent=[x, x + w, y + h, y],
            )
        for site, grade, bbox in osp_results["detections"]:
            if grade > 0:
                color = SEVERITY_COLORS.get(grade, "white")
                x, y, w, h = bbox
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=2, edgecolor=color,
                    facecolor="none", linestyle="--",
                )
                ax.add_patch(rect)
                ax.text(
                    x, y - 3,
                    f"{site}: G{grade}",
                    color=color, fontsize=8, weight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", alpha=0.6),
                )

    # Layer 3: Sclerosis heatmap
    if show_sclerosis and "heatmap" in scl_results and scl_results["heatmap"] is not None:
        heatmap = scl_results["heatmap"]
        extent = scl_results.get("roi_extent", [0, image.shape[1], image.shape[0], 0])
        ax.imshow(heatmap, cmap="YlOrRd", alpha=0.3, extent=extent)

    # KL Grade badge
    grade = kl_pred.get("grade", "?")
    confidence = kl_pred.get("confidence", 0.0)
    ax.text(
        5, 15,
        f"KL Grade: {grade} ({confidence:.0%})",
        fontsize=13, weight="bold", color="white",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="black", alpha=0.8),
    )

    ax.set_axis_off()
    fig.tight_layout(pad=0.5)
    return fig


def create_jsn_measurement_pairs(
    femoral: np.ndarray,
    tibial: np.ndarray,
    n_lines: int = 16,
) -> List[Tuple]:
    """Create measurement line pairs for JSN visualization.

    Returns:
        List of (point1, point2, distance) tuples for drawing.
    """
    if femoral is None or tibial is None:
        return []
    _, pairs = compute_measurement_pairs_from_contours(
        femoral,
        tibial,
        n_points=n_lines,
        smoothing_window=5,
    )
    return pairs
