"""Automatic subchondral bone ROI extraction from JSN segmentation masks."""

from typing import Optional, Tuple

import cv2
import numpy as np


def get_tibial_surface(mask: np.ndarray, compartment_label: Optional[int] = None) -> Optional[np.ndarray]:
    """Extract the tibial plateau surface contour from JSN segmentation mask.

    The tibial surface is the lower boundary of the joint space region.

    Args:
        mask: (H, W) segmentation mask with 0=bg, 1=medial JS, 2=lateral JS.
        compartment_label: optional mask label to isolate one compartment.

    Returns:
        (N, 2) array of (x, y) surface points, or None if not found.
    """
    if compartment_label is None:
        js_mask = ((mask == 1) | (mask == 2)).astype(np.uint8)
    else:
        js_mask = (mask == int(compartment_label)).astype(np.uint8)

    if js_mask.sum() < 10:
        return None

    surface_points = []
    for x in range(mask.shape[1]):
        col = js_mask[:, x]
        nonzero = np.where(col > 0)[0]
        if len(nonzero) > 0:
            surface_points.append([x, nonzero[-1]])  # bottommost = tibial surface

    if len(surface_points) < 5:
        return None

    return np.array(surface_points, dtype=np.float64)


def extract_subchondral_roi(
    mask: np.ndarray,
    image: np.ndarray,
    depth_px: int = 10,
    offset_pct: float = 0.10,
    medial_offset_pct: Optional[float] = None,
    lateral_offset_pct: Optional[float] = None,
    is_left: bool = False,
    depth_fraction: float = 0.10,
    medial_depth_fraction: Optional[float] = None,
    lateral_depth_fraction: Optional[float] = None,
    medial_inner_offset_pct: Optional[float] = None,
    medial_outer_offset_pct: Optional[float] = None,
    lateral_inner_offset_pct: Optional[float] = None,
    lateral_outer_offset_pct: Optional[float] = None,
    surface_offset_fraction: float = 0.015,
    surface_smoothing_window: int = 7,
    output_size: int = 64,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    medial_roi, lateral_roi, _, _ = extract_subchondral_roi_with_boxes(
        mask=mask,
        image=image,
        depth_px=depth_px,
        offset_pct=offset_pct,
        medial_offset_pct=medial_offset_pct,
        lateral_offset_pct=lateral_offset_pct,
        is_left=is_left,
        depth_fraction=depth_fraction,
        medial_depth_fraction=medial_depth_fraction,
        lateral_depth_fraction=lateral_depth_fraction,
        medial_inner_offset_pct=medial_inner_offset_pct,
        medial_outer_offset_pct=medial_outer_offset_pct,
        lateral_inner_offset_pct=lateral_inner_offset_pct,
        lateral_outer_offset_pct=lateral_outer_offset_pct,
        surface_offset_fraction=surface_offset_fraction,
        surface_smoothing_window=surface_smoothing_window,
        output_size=output_size,
    )
    return medial_roi, lateral_roi


def extract_subchondral_roi_with_boxes(
    mask: np.ndarray,
    image: np.ndarray,
    depth_px: int = 10,
    offset_pct: float = 0.10,
    medial_offset_pct: Optional[float] = None,
    lateral_offset_pct: Optional[float] = None,
    is_left: bool = False,
    depth_fraction: float = 0.10,
    medial_depth_fraction: Optional[float] = None,
    lateral_depth_fraction: Optional[float] = None,
    medial_inner_offset_pct: Optional[float] = None,
    medial_outer_offset_pct: Optional[float] = None,
    lateral_inner_offset_pct: Optional[float] = None,
    lateral_outer_offset_pct: Optional[float] = None,
    surface_offset_fraction: float = 0.015,
    surface_smoothing_window: int = 7,
    output_size: int = 64,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[Tuple[int, int, int, int]],
    Optional[Tuple[int, int, int, int]],
]:
    medial_roi, lateral_roi, medial_box, lateral_box, _ = extract_subchondral_roi_with_boxes_and_source(
        mask=mask,
        image=image,
        depth_px=depth_px,
        offset_pct=offset_pct,
        medial_offset_pct=medial_offset_pct,
        lateral_offset_pct=lateral_offset_pct,
        is_left=is_left,
        depth_fraction=depth_fraction,
        medial_depth_fraction=medial_depth_fraction,
        lateral_depth_fraction=lateral_depth_fraction,
        medial_inner_offset_pct=medial_inner_offset_pct,
        medial_outer_offset_pct=medial_outer_offset_pct,
        lateral_inner_offset_pct=lateral_inner_offset_pct,
        lateral_outer_offset_pct=lateral_outer_offset_pct,
        surface_offset_fraction=surface_offset_fraction,
        surface_smoothing_window=surface_smoothing_window,
        output_size=output_size,
    )
    return medial_roi, lateral_roi, medial_box, lateral_box


def extract_subchondral_roi_with_boxes_and_source(
    mask: np.ndarray,
    image: np.ndarray,
    depth_px: int = 10,
    offset_pct: float = 0.10,
    medial_offset_pct: Optional[float] = None,
    lateral_offset_pct: Optional[float] = None,
    is_left: bool = False,
    depth_fraction: float = 0.10,
    medial_depth_fraction: Optional[float] = None,
    lateral_depth_fraction: Optional[float] = None,
    medial_inner_offset_pct: Optional[float] = None,
    medial_outer_offset_pct: Optional[float] = None,
    lateral_inner_offset_pct: Optional[float] = None,
    lateral_outer_offset_pct: Optional[float] = None,
    surface_offset_fraction: float = 0.015,
    surface_smoothing_window: int = 7,
    output_size: int = 64,
) -> Tuple[
    Optional[np.ndarray],
    Optional[np.ndarray],
    Optional[Tuple[int, int, int, int]],
    Optional[Tuple[int, int, int, int]],
    str,
]:
    """Extract medial and lateral subchondral bone ROI patches.

    Extracts the bone region immediately below the tibial plateau surface,
    using the JSN segmentation mask to identify the surface location.

    Args:
        mask: (H, W) JSN segmentation mask.
        image: (H, W) grayscale knee X-ray image.
        depth_px: Minimum depth into bone below tibial surface (pixels).
        offset_pct: Default lateral offset percentage to avoid osteophyte contamination.
        medial_offset_pct: Optional medial-compartment offset override.
        lateral_offset_pct: Optional lateral-compartment offset override.
        is_left: True if this is a left knee image (flips medial/lateral assignment).
        depth_fraction: Relative band depth as a fraction of image height.
        medial_depth_fraction: Optional medial depth override as a fraction of image height.
        lateral_depth_fraction: Optional lateral depth override as a fraction of image height.
        medial_inner_offset_pct: Optional trim near the joint-center side of the medial compartment.
        medial_outer_offset_pct: Optional trim near the cortical-edge side of the medial compartment.
        lateral_inner_offset_pct: Optional trim near the joint-center side of the lateral compartment.
        lateral_outer_offset_pct: Optional trim near the cortical-edge side of the lateral compartment.
        surface_offset_fraction: Small downward offset from the tibial surface.
        surface_smoothing_window: Smoothing window for the surface profile.
        output_size: Output patch size for downstream CNN input.

    Laterality convention (AP view):
        Right knee: medial = right half (x >= midline), lateral = left half
        Left knee:  medial = left half  (x < midline),  lateral = right half

    Returns:
        (medial_roi, lateral_roi, medial_box, lateral_box, source_mode)
        where source_mode describes the JSN guidance path used.
    """
    medial_pts = get_tibial_surface(mask, compartment_label=1)
    lateral_pts = get_tibial_surface(mask, compartment_label=2)
    source_mode = "jsn_guided"

    h, w = image.shape[:2]
    midline_x = w // 2

    if medial_pts is None or lateral_pts is None:
        tibial_surface = get_tibial_surface(mask)
        if tibial_surface is None:
            return None, None, None, None, "jsn_missing"
        source_mode = "jsn_midline_split"

        # Legacy fallback when a predicted mask is missing one compartment label.
        if is_left:
            medial_pts = tibial_surface[tibial_surface[:, 0] < midline_x]
            lateral_pts = tibial_surface[tibial_surface[:, 0] >= midline_x]
        else:
            medial_pts = tibial_surface[tibial_surface[:, 0] >= midline_x]
            lateral_pts = tibial_surface[tibial_surface[:, 0] < midline_x]

    if medial_offset_pct is None:
        medial_offset_pct = offset_pct
    if lateral_offset_pct is None:
        lateral_offset_pct = offset_pct
    if medial_depth_fraction is None:
        medial_depth_fraction = depth_fraction
    if lateral_depth_fraction is None:
        lateral_depth_fraction = depth_fraction

    medial_left_trim, medial_right_trim = _resolve_trim_offsets(
        compartment="medial",
        is_left=is_left,
        default_offset=float(medial_offset_pct),
        inner_offset=medial_inner_offset_pct,
        outer_offset=medial_outer_offset_pct,
    )
    lateral_left_trim, lateral_right_trim = _resolve_trim_offsets(
        compartment="lateral",
        is_left=is_left,
        default_offset=float(lateral_offset_pct),
        inner_offset=lateral_inner_offset_pct,
        outer_offset=lateral_outer_offset_pct,
    )

    medial_roi, medial_box = _crop_below_surface(
        image=image,
        surface_pts=medial_pts,
        min_depth_px=depth_px,
        left_trim_pct=medial_left_trim,
        right_trim_pct=medial_right_trim,
        h=h,
        w=w,
        depth_fraction=float(medial_depth_fraction),
        surface_offset_fraction=surface_offset_fraction,
        surface_smoothing_window=surface_smoothing_window,
        output_size=output_size,
    )
    lateral_roi, lateral_box = _crop_below_surface(
        image=image,
        surface_pts=lateral_pts,
        min_depth_px=depth_px,
        left_trim_pct=lateral_left_trim,
        right_trim_pct=lateral_right_trim,
        h=h,
        w=w,
        depth_fraction=float(lateral_depth_fraction),
        surface_offset_fraction=surface_offset_fraction,
        surface_smoothing_window=surface_smoothing_window,
        output_size=output_size,
    )

    return medial_roi, lateral_roi, medial_box, lateral_box, source_mode


def _smooth_series(values: np.ndarray, kernel_size: int = 7) -> np.ndarray:
    if values.size < 3:
        return values
    kernel_size = max(3, int(kernel_size))
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = min(kernel_size, values.size if values.size % 2 == 1 else values.size - 1)
    if kernel_size < 3:
        return values
    kernel = np.ones(kernel_size, dtype=np.float64) / kernel_size
    padded = np.pad(values, (kernel_size // 2, kernel_size // 2), mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def _crop_below_surface(
    image: np.ndarray,
    surface_pts: np.ndarray,
    min_depth_px: int,
    left_trim_pct: float,
    right_trim_pct: float,
    h: int,
    w: int,
    depth_fraction: float,
    surface_offset_fraction: float,
    surface_smoothing_window: int,
    output_size: int,
) -> Optional[np.ndarray]:
    """Crop a contour-following band below the tibial surface."""
    if len(surface_pts) < 3:
        return None, None

    order = np.argsort(surface_pts[:, 0])
    ordered = np.asarray(surface_pts[order], dtype=np.float64)
    unique_x, unique_idx = np.unique(ordered[:, 0], return_index=True)
    ordered = ordered[unique_idx]
    if len(ordered) < 3:
        return None, None

    x_min = int(surface_pts[:, 0].min())
    x_max = int(surface_pts[:, 0].max())
    x_span = x_max - x_min

    left_trim = int(round(x_span * max(0.0, float(left_trim_pct))))
    right_trim = int(round(x_span * max(0.0, float(right_trim_pct))))
    x_min = min(x_min + left_trim, w - 1)
    x_max = max(x_max - right_trim, 0)

    if x_max <= x_min:
        return None, None

    relevant = ordered[(ordered[:, 0] >= x_min) & (ordered[:, 0] <= x_max)]
    if len(relevant) < 3:
        return None, None

    x_coords = np.arange(x_min, x_max + 1, dtype=np.int32)
    if x_coords.size < 4:
        return None, None

    surface_y = np.interp(x_coords, relevant[:, 0], relevant[:, 1])
    surface_y = _smooth_series(surface_y, surface_smoothing_window)

    depth = max(int(min_depth_px), int(round(depth_fraction * h)))
    depth = max(depth, 4)
    y_offset = max(2, int(round(surface_offset_fraction * h)))
    top_profile = np.clip(np.round(surface_y + y_offset).astype(np.int32), 0, h - 2)
    bottom_profile = np.clip(top_profile + depth, 0, h)

    band = np.zeros((depth, x_coords.size), dtype=image.dtype)
    valid_columns = 0
    for idx, (x_val, y_top, y_bottom) in enumerate(zip(x_coords, top_profile, bottom_profile)):
        if y_bottom <= y_top + 1:
            continue
        column = image[y_top:y_bottom, x_val]
        if column.size == 0:
            continue
        valid_columns += 1
        if column.size < depth:
            pad_value = column[-1]
            padded = np.full(depth, pad_value, dtype=image.dtype)
            padded[:column.size] = column
            column = padded
        elif column.size > depth:
            column = column[:depth]
        band[:, idx] = column

    if valid_columns < max(4, x_coords.size // 3):
        return None, None

    roi = cv2.resize(band, (output_size, output_size), interpolation=cv2.INTER_LINEAR)
    box = (
        int(x_coords[0]),
        int(top_profile.min()),
        int(x_coords[-1]) + 1,
        int(bottom_profile.max()),
    )
    return roi, box


def _resolve_trim_offsets(
    compartment: str,
    is_left: bool,
    default_offset: float,
    inner_offset: Optional[float],
    outer_offset: Optional[float],
) -> Tuple[float, float]:
    """Map inner/outer anatomical trims to left/right x-direction trims."""
    inner = float(default_offset if inner_offset is None else inner_offset)
    outer = float(default_offset if outer_offset is None else outer_offset)

    if compartment == "medial":
        if is_left:
            return outer, inner
        return inner, outer

    if compartment == "lateral":
        if is_left:
            return inner, outer
        return outer, inner

    return float(default_offset), float(default_offset)
