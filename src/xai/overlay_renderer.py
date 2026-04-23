"""Multi-layer XAI overlay renderer for annotated X-ray visualization."""

from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

from src.features.jsw_computation import compute_measurement_pairs_from_contours


SEVERITY_COLORS = {0: None, 1: "#22c55e", 2: "#f59e0b", 3: "#ef4444"}
SEVERITY_LABELS = {0: "None", 1: "Small", 2: "Moderate", 3: "Large"}
SITE_LABELS = {
    "medial_femur": "MF",
    "lateral_femur": "LF",
    "medial_tibia": "MT",
    "lateral_tibia": "LT",
}


def generate_xai_overlay(
    image: np.ndarray,
    jsn_results: Dict,
    osp_results: Dict,
    scl_results: Dict,
    kl_pred: Dict,
    show_jsn: bool = True,
    show_jsn_medial: bool = True,
    show_jsn_lateral: bool = True,
    show_osteophytes: bool = True,
    show_sclerosis: bool = True,
    show_kl_badge: bool = True,
    distance_units: str = "px",
    pixel_spacing_mm: Optional[float] = None,
    figsize: Tuple[int, int] = (7, 7),
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
    fig, ax = plt.subplots(1, 1, figsize=figsize, facecolor="#0f172a")
    ax.imshow(image, cmap="gray")
    height, width = image.shape[:2]

    # Layer 1: closest JSN area per compartment
    if show_jsn:
        if show_jsn_lateral:
            _draw_closest_jsn_area(
                ax,
                jsn_results.get("measurement_pairs_lateral", []),
                "Lateral JSN",
                "#38bdf8",
                distance_units=distance_units,
                pixel_spacing_mm=pixel_spacing_mm,
            )
        if show_jsn_medial:
            _draw_closest_jsn_area(
                ax,
                jsn_results.get("measurement_pairs_medial", []),
                "Medial JSN",
                "#22d3ee",
                distance_units=distance_units,
                pixel_spacing_mm=pixel_spacing_mm,
            )

    # Layer 2: Osteophyte severity boxes
    if show_osteophytes and "detections" in osp_results:
        for site, grade, bbox in osp_results["detections"]:
            if grade > 0:
                color = SEVERITY_COLORS.get(grade, "white")
                x, y, w, h = bbox
                rect = patches.Rectangle(
                    (x, y), w, h,
                    linewidth=1.6, edgecolor=color,
                    facecolor="none", linestyle="-",
                )
                ax.add_patch(rect)
                ax.text(
                    x + 2,
                    y + 7,
                    f"{SITE_LABELS.get(site, site)} G{grade}",
                    color="#ffffff",
                    fontsize=6,
                    weight="bold",
                    clip_on=True,
                    bbox=dict(boxstyle="round,pad=0.16", fc=color, ec="none", alpha=0.88),
                )

    # Layer 3: Sclerosis heatmap
    if show_sclerosis and "heatmap" in scl_results and scl_results["heatmap"] is not None:
        heatmap = scl_results["heatmap"]
        extent = scl_results.get("roi_extent", [0, image.shape[1], image.shape[0], 0])
        ax.imshow(heatmap, cmap="YlOrRd", alpha=0.22, extent=extent)

    # KL Grade badge
    if show_kl_badge:
        grade = kl_pred.get("grade", "?")
        confidence = kl_pred.get("confidence", 0.0)
        ax.text(
            0.025,
            0.045,
            f"KL {grade}  |  {confidence:.0%}",
            transform=ax.transAxes,
            fontsize=10,
            weight="bold",
            color="#ffffff",
            bbox=dict(boxstyle="round,pad=0.35", facecolor="#111827", edgecolor="#334155", alpha=0.88),
        )
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_axis_off()
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    return fig


def _draw_closest_jsn_area(
    ax,
    pairs,
    label: str,
    color: str,
    distance_units: str = "px",
    pixel_spacing_mm: Optional[float] = None,
) -> None:
    if not pairs:
        return
    p1, p2, dist = min(pairs, key=lambda item: item[2])
    p1 = np.asarray(p1, dtype=np.float64)
    p2 = np.asarray(p2, dtype=np.float64)
    vector = p2 - p1
    length = float(np.linalg.norm(vector))
    if length < 1e-6:
        return
    normal = np.asarray([-vector[1], vector[0]], dtype=np.float64) / length
    half_width = max(4.0, min(12.0, length * 0.45))
    polygon = np.vstack([
        p1 + normal * half_width,
        p2 + normal * half_width,
        p2 - normal * half_width,
        p1 - normal * half_width,
    ])
    patch = patches.Polygon(
        polygon,
        closed=True,
        facecolor=color,
        edgecolor=color,
        linewidth=1.6,
        alpha=0.30,
    )
    ax.add_patch(patch)
    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=2.0, alpha=0.95)
    midpoint = (p1 + p2) / 2.0
    distance_label = _format_distance(dist, distance_units=distance_units, pixel_spacing_mm=pixel_spacing_mm)
    ax.text(
        float(midpoint[0]),
        float(midpoint[1]),
        f"{label}: {distance_label}",
        color="#ffffff",
        fontsize=6.2,
        weight="bold",
        ha="center",
        va="center",
        clip_on=True,
        bbox=dict(boxstyle="round,pad=0.18", fc="#0f172a", ec=color, alpha=0.78),
    )


def _format_distance(dist_px: float, distance_units: str = "px", pixel_spacing_mm: Optional[float] = None) -> str:
    units = str(distance_units or "px").strip().lower()
    if units == "mm":
        if pixel_spacing_mm is not None and pixel_spacing_mm > 0:
            return f"{dist_px * pixel_spacing_mm:.1f} mm"
        return f"{dist_px:.1f} px"
    if units == "both":
        if pixel_spacing_mm is not None and pixel_spacing_mm > 0:
            return f"{dist_px * pixel_spacing_mm:.1f} mm / {dist_px:.1f} px"
        return f"{dist_px:.1f} px"
    return f"{dist_px:.1f} px"


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
