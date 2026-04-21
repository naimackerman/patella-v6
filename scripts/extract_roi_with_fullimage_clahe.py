"""Extract osteophyte ROIs after applying CLAHE to the full image.

Best-practice path for the osteophyte stage:
1. Enhance the full radiograph with CLAHE.
2. Use reviewed JSN masks when available, or JSN segmenter masks otherwise.
3. Fall back to heuristic landmarks, then fixed geometric crops if needed.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
from pathlib import Path
import sys
import time

import cv2
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.features.kneel_landmarks import KNEELLandmarkDetector, extract_kneel_rois, landmarks_from_jsn_mask
from src.models.roi_detector import ROIDetector
from src.data.image_validation import read_grayscale_image


ROI_SITES = ["medial_femur", "lateral_femur", "medial_tibia", "lateral_tibia"]
DEFAULT_OSTEOPHYTE_ROI_SIZE = 140


def apply_clahe_full_image(image: np.ndarray, clip_limit: float = 3.0, tile_grid_size: int = 8) -> np.ndarray:
    """Apply CLAHE on the full image before ROI extraction."""
    clahe = cv2.createCLAHE(
        clipLimit=float(clip_limit),
        tileGridSize=(int(tile_grid_size), int(tile_grid_size)),
    )
    return clahe.apply(image)


def extract_geometric_roi(
    image: np.ndarray,
    is_left: bool,
    site: str,
    roi_size: int = DEFAULT_OSTEOPHYTE_ROI_SIZE,
) -> np.ndarray:
    """Geometric osteophyte ROI fallback used when landmark extraction fails."""
    h, w = image.shape

    if is_left:
        if site == "medial_femur":
            x1, y1, x2, y2 = int(w * 0.05), int(h * 0.15), int(w * 0.45), int(h * 0.45)
        elif site == "lateral_femur":
            x1, y1, x2, y2 = int(w * 0.55), int(h * 0.15), int(w * 0.95), int(h * 0.45)
        elif site == "medial_tibia":
            x1, y1, x2, y2 = int(w * 0.05), int(h * 0.55), int(w * 0.45), int(h * 0.85)
        else:
            x1, y1, x2, y2 = int(w * 0.55), int(h * 0.55), int(w * 0.95), int(h * 0.85)
    else:
        if site == "medial_femur":
            x1, y1, x2, y2 = int(w * 0.55), int(h * 0.15), int(w * 0.95), int(h * 0.45)
        elif site == "lateral_femur":
            x1, y1, x2, y2 = int(w * 0.05), int(h * 0.15), int(w * 0.45), int(h * 0.45)
        elif site == "medial_tibia":
            x1, y1, x2, y2 = int(w * 0.55), int(h * 0.55), int(w * 0.95), int(h * 0.85)
        else:
            x1, y1, x2, y2 = int(w * 0.05), int(h * 0.55), int(w * 0.45), int(h * 0.85)

    roi = image[y1:y2, x1:x2]
    if roi.size == 0:
        return np.zeros((roi_size, roi_size), dtype=np.uint8)
    return cv2.resize(roi, (roi_size, roi_size), interpolation=cv2.INTER_LINEAR)


def extract_geometric_rois(
    image: np.ndarray,
    is_left: bool,
    roi_size: int = DEFAULT_OSTEOPHYTE_ROI_SIZE,
) -> dict[str, np.ndarray]:
    """Extract all osteophyte ROIs using the fixed geometric fallback."""
    return {site: extract_geometric_roi(image, is_left, site, roi_size=roi_size) for site in ROI_SITES}


def extract_landmark_rois(
    image: np.ndarray,
    is_left: bool,
    landmark_detector: KNEELLandmarkDetector,
    roi_size: int = DEFAULT_OSTEOPHYTE_ROI_SIZE,
) -> dict[str, np.ndarray]:
    """Extract all osteophyte ROIs using landmark-guided boxes."""
    boxes = ROIDetector.landmark_boxes(
        image,
        is_left=is_left,
        landmark_detector=landmark_detector,
        apply_preprocessing=False,
        require_reliable=True,
    )
    rois = ROIDetector.crop_from_boxes(image, boxes, osteophyte_roi_size=roi_size)
    return {site: rois[site] for site in ROI_SITES}


def _landmark_worker(
    queue: mp.queues.Queue,
    image: np.ndarray,
    is_left: bool,
    backend: str,
    roi_size: int,
) -> None:
    try:
        detector = KNEELLandmarkDetector(
            backend=backend,
            allow_backend_fallback=True,
        )
        rois = extract_landmark_rois(image, is_left, detector, roi_size=roi_size)
        queue.put(("ok", rois))
    except Exception as exc:  # pragma: no cover - defensive worker boundary
        queue.put(("error", f"{type(exc).__name__}:{exc}"))


def extract_landmark_rois_with_timeout(
    image: np.ndarray,
    is_left: bool,
    backend: str,
    timeout_seconds: float,
    roi_size: int = DEFAULT_OSTEOPHYTE_ROI_SIZE,
) -> dict[str, np.ndarray]:
    if timeout_seconds <= 0:
        detector = KNEELLandmarkDetector(
            backend=backend,
            allow_backend_fallback=True,
        )
        return extract_landmark_rois(image, is_left, detector, roi_size=roi_size)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(
        target=_landmark_worker,
        args=(queue, image, is_left, backend, roi_size),
    )
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        raise TimeoutError(f"landmark_timeout>{timeout_seconds:.1f}s")

    if queue.empty():
        raise RuntimeError("landmark_worker_no_result")

    status, payload = queue.get()
    if status == "ok":
        return payload
    raise RuntimeError(str(payload))


def _image_read_worker(queue: mp.queues.Queue, path: str) -> None:
    image = read_grayscale_image(path)
    queue.put(image)


def read_grayscale_image_with_timeout(path: Path, timeout_seconds: float) -> np.ndarray | None:
    if timeout_seconds <= 0:
        return read_grayscale_image(path)

    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    process = ctx.Process(target=_image_read_worker, args=(queue, str(path)))
    process.start()
    process.join(timeout_seconds)

    if process.is_alive():
        process.terminate()
        process.join()
        return None

    if queue.empty():
        return None
    return queue.get()


def load_jsn_mask(mask_path: Path) -> np.ndarray | None:
    if mask_path.suffix == ".npy":
        try:
            return np.load(str(mask_path)).astype(np.uint8)
        except Exception:
            return None
    return read_grayscale_image(mask_path)


def resolve_jsn_mask_path(
    reviewed_mask_dir: Path | None,
    predicted_mask_dir: Path | None,
    split: str,
    image_id: str,
) -> tuple[Path | None, str | None]:
    if reviewed_mask_dir is not None:
        reviewed_path = reviewed_mask_dir / split / f"{image_id}.png"
        if reviewed_path.exists():
            return reviewed_path, "jsn_reviewed"
    if predicted_mask_dir is not None:
        predicted_path = predicted_mask_dir / f"{image_id}_mask.npy"
        if predicted_path.exists():
            return predicted_path, "jsn_predicted"
    return None, None


def extract_jsn_guided_rois(
    image: np.ndarray,
    mask: np.ndarray,
    roi_size: int = DEFAULT_OSTEOPHYTE_ROI_SIZE,
) -> dict[str, np.ndarray]:
    landmarks = landmarks_from_jsn_mask(mask)
    if landmarks.low_confidence:
        raise ValueError("jsn_mask_low_confidence")
    rois = extract_kneel_rois(image, landmarks, osteophyte_roi_size=roi_size)
    return {site: rois[site] for site in ROI_SITES}


def resolve_failure_log_path(output_dir: Path, failure_log: str | None) -> Path:
    if failure_log:
        return Path(failure_log)
    return output_dir / "roi_extraction_failures.csv"


def resolve_audit_log_path(output_dir: Path, audit_log: str | None) -> Path:
    if audit_log:
        return Path(audit_log)
    return output_dir / "roi_extraction_audit.csv"


def build_dataset_index(input_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, str]] = []
    for split_dir in sorted(input_dir.iterdir()):
        if not split_dir.is_dir():
            continue
        split = split_dir.name
        if split not in {"train", "val", "test"}:
            continue
        for grade_dir in sorted(split_dir.iterdir()):
            if not grade_dir.is_dir():
                continue
            for image_path in sorted(grade_dir.glob("*.png")):
                rows.append({
                    "image_id": image_path.stem,
                    "split": split,
                })
    return pd.DataFrame(rows)


def load_retry_index(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"image_id", "split"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Retry CSV is missing required columns: {sorted(missing)}")
    subset = df.loc[:, ["image_id", "split"]].copy()
    subset["image_id"] = subset["image_id"].astype(str).str.replace(".png", "", regex=False)
    subset["split"] = subset["split"].astype(str)
    subset = subset.drop_duplicates().reset_index(drop=True)
    return subset


def image_has_all_roi_outputs(output_dir: Path, split: str, image_id: str, roi_size: int) -> bool:
    split_dir = output_dir / split
    for site in ROI_SITES:
        roi_path = split_dir / f"{image_id}_{site}.png"
        if not roi_path.exists():
            return False
        roi = cv2.imread(str(roi_path), cv2.IMREAD_GRAYSCALE)
        if roi is None or roi.shape[:2] != (roi_size, roi_size):
            return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=str,
        default="KneeXrayData/Dataset",
        help="Input directory with train/val/test/<grade> image folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="features/rois_osteophyte_clahe_full",
        help="Output directory for extracted osteophyte ROI patches.",
    )
    parser.add_argument(
        "--labels-csv",
        type=str,
        default=None,
        help="Optional CSV providing the image_id/split list to process.",
    )
    parser.add_argument(
        "--scan-all-images",
        action="store_true",
        help="Ignore labels CSV and scan the full input dataset tree.",
    )
    parser.add_argument(
        "--retry-failures-csv",
        type=str,
        default=None,
        help="Retry only image_id/split rows from a previous failure CSV.",
    )
    parser.add_argument("--clahe-clip", type=float, default=3.0, help="CLAHE clip limit.")
    parser.add_argument("--clahe-tile", type=int, default=8, help="CLAHE tile grid size.")
    parser.add_argument(
        "--osteophyte-roi-size",
        type=int,
        default=DEFAULT_OSTEOPHYTE_ROI_SIZE,
        help="Output size in pixels for each square osteophyte ROI patch.",
    )
    parser.add_argument(
        "--landmark-backend",
        choices=("heuristic", "kneel_repo"),
        default="heuristic",
        help="Landmark backend to use for ROI extraction.",
    )
    parser.add_argument(
        "--geometry-only",
        action="store_true",
        help="Disable landmark extraction and use geometric ROIs only.",
    )
    parser.add_argument(
        "--failure-log",
        type=str,
        default=None,
        help="Optional CSV path for logging landmark failures and missing images.",
    )
    parser.add_argument(
        "--audit-log",
        type=str,
        default=None,
        help="Optional CSV path for logging per-image extraction mode.",
    )
    parser.add_argument(
        "--reviewed-jsn-mask-dir",
        type=str,
        default="annotations/jsn_masks",
        help="Directory containing reviewed JSN masks under split subfolders.",
    )
    parser.add_argument(
        "--predicted-jsn-mask-dir",
        type=str,
        default="features/jsn/masks",
        help="Directory containing predicted JSN masks named <image_id>_mask.npy.",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=25,
        help="Print a durable progress line every N processed images.",
    )
    parser.add_argument(
        "--landmark-timeout-seconds",
        type=float,
        default=5.0,
        help="Maximum time to allow heuristic landmark ROI extraction before geometric fallback.",
    )
    parser.add_argument(
        "--skip-landmark-if-no-jsn",
        action="store_true",
        help="When no reviewed/predicted JSN mask exists, skip heuristic landmarks and use geometric fallback directly.",
    )
    parser.add_argument(
        "--image-read-timeout-seconds",
        type=float,
        default=5.0,
        help="Maximum time to allow image decoding before treating the image as unreadable.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip images whose 4 ROI patch outputs already exist in the output directory.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    osteophyte_roi_size = int(args.osteophyte_roi_size)
    if osteophyte_roi_size <= 0:
        raise ValueError(f"--osteophyte-roi-size must be positive, got {osteophyte_roi_size}")
    failure_log_path = resolve_failure_log_path(output_dir, args.failure_log)
    audit_log_path = resolve_audit_log_path(output_dir, args.audit_log)

    print("=" * 70, flush=True)
    print("ROI EXTRACTION WITH FULL-IMAGE CLAHE", flush=True)
    print("=" * 70, flush=True)
    print(f"Input: {args.input_dir}", flush=True)
    print(f"Output: {args.output_dir}", flush=True)
    if args.retry_failures_csv:
        label_desc = f"retry failures from {args.retry_failures_csv}"
    else:
        label_desc = "scan full dataset" if args.scan_all_images else (args.labels_csv or "none")
    print(f"Labels: {label_desc}", flush=True)
    print(f"CLAHE: clip_limit={args.clahe_clip}, tile_grid={args.clahe_tile}x{args.clahe_tile}", flush=True)
    print(f"Osteophyte ROI size: {osteophyte_roi_size}x{osteophyte_roi_size}", flush=True)
    if args.geometry_only:
        roi_mode = "geometric only"
    else:
        roi_mode = f"JSN-guided (reviewed/predicted) -> landmark ({args.landmark_backend}) -> geometric fallback"
    print(f"ROI mode: {roi_mode}", flush=True)
    print(flush=True)

    input_dir = Path(args.input_dir)
    if args.retry_failures_csv:
        labels_df = load_retry_index(Path(args.retry_failures_csv))
    elif args.scan_all_images:
        labels_df = build_dataset_index(input_dir)
    else:
        if not args.labels_csv:
            raise ValueError("Provide --labels-csv or use --scan-all-images.")
        labels_df = pd.read_csv(args.labels_csv)
    print(f"Total images to process: {len(labels_df)}", flush=True)

    for split in ("train", "val", "test"):
        (output_dir / split).mkdir(parents=True, exist_ok=True)

    reviewed_jsn_mask_dir = Path(args.reviewed_jsn_mask_dir) if args.reviewed_jsn_mask_dir else None
    predicted_jsn_mask_dir = Path(args.predicted_jsn_mask_dir) if args.predicted_jsn_mask_dir else None

    total_images = len(labels_df)
    mode_counts = {
        "jsn_reviewed": 0,
        "jsn_predicted": 0,
        "landmark": 0,
        "geometric_fallback": 0,
    }
    processed = 0
    skipped_existing = 0
    started_at = time.monotonic()
    progress_every = max(1, int(args.progress_every))

    audit_log_path.parent.mkdir(parents=True, exist_ok=True)
    failure_log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(audit_log_path, "w", newline="", encoding="utf-8") as audit_handle, open(
        failure_log_path, "w", newline="", encoding="utf-8"
    ) as failure_handle:
        audit_writer = csv.DictWriter(audit_handle, fieldnames=["image_id", "split", "mode", "backend", "reason"])
        failure_writer = csv.DictWriter(failure_handle, fieldnames=["image_id", "split", "reason"])
        audit_writer.writeheader()
        failure_writer.writeheader()

        for index, (_, row) in enumerate(labels_df.iterrows(), start=1):
            image_id = str(row["image_id"]).replace(".png", "")
            split = str(row["split"])
            is_left = image_id.upper().endswith("L")

            if args.skip_existing and image_has_all_roi_outputs(output_dir, split, image_id, osteophyte_roi_size):
                skipped_existing += 1
                if index % progress_every == 0 or index == total_images:
                    elapsed = time.monotonic() - started_at
                    print(
                        f"[ROI] {index}/{total_images} images, "
                        f"jsn_reviewed={mode_counts['jsn_reviewed']}, "
                        f"jsn_predicted={mode_counts['jsn_predicted']}, "
                        f"landmark={mode_counts['landmark']}, "
                        f"fallback={mode_counts['geometric_fallback']}, "
                        f"skipped_existing={skipped_existing}, "
                        f"last={image_id}:skipped_existing, "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    audit_handle.flush()
                    failure_handle.flush()
                continue

            image_path = None
            for grade in range(5):
                candidate = input_dir / split / str(grade) / f"{image_id}.png"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path is None:
                failure_writer.writerow({"image_id": image_id, "split": split, "reason": "image_not_found"})
                audit_writer.writerow({
                    "image_id": image_id,
                    "split": split,
                    "mode": "missing_image",
                    "backend": args.landmark_backend if not args.geometry_only else "none",
                    "reason": "image_not_found",
                })
                if index % progress_every == 0:
                    elapsed = time.monotonic() - started_at
                    print(
                        f"[ROI] {index}/{total_images} images, "
                        f"jsn_reviewed={mode_counts['jsn_reviewed']}, "
                        f"jsn_predicted={mode_counts['jsn_predicted']}, "
                        f"landmark={mode_counts['landmark']}, "
                        f"fallback={mode_counts['geometric_fallback']}, "
                        f"skipped_existing={skipped_existing}, "
                        f"last={image_id}:missing_image, "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    audit_handle.flush()
                    failure_handle.flush()
                continue

            image = read_grayscale_image_with_timeout(
                image_path,
                timeout_seconds=float(args.image_read_timeout_seconds),
            )
            if image is None:
                failure_writer.writerow({"image_id": image_id, "split": split, "reason": "image_load_failed"})
                audit_writer.writerow({
                    "image_id": image_id,
                    "split": split,
                    "mode": "load_failed",
                    "backend": args.landmark_backend if not args.geometry_only else "none",
                    "reason": "image_load_failed",
                })
                if index % progress_every == 0:
                    elapsed = time.monotonic() - started_at
                    print(
                        f"[ROI] {index}/{total_images} images, "
                        f"jsn_reviewed={mode_counts['jsn_reviewed']}, "
                        f"jsn_predicted={mode_counts['jsn_predicted']}, "
                        f"landmark={mode_counts['landmark']}, "
                        f"fallback={mode_counts['geometric_fallback']}, "
                        f"skipped_existing={skipped_existing}, "
                        f"last={image_id}:load_failed, "
                        f"elapsed={elapsed:.1f}s",
                        flush=True,
                    )
                    audit_handle.flush()
                    failure_handle.flush()
                continue

            enhanced = apply_clahe_full_image(image, clip_limit=args.clahe_clip, tile_grid_size=args.clahe_tile)

            extraction_mode = "geometric_fallback"
            reason = "geometry_only" if args.geometry_only else "landmark_success"
            rois = None
            jsn_mask_path, jsn_mode = resolve_jsn_mask_path(
                reviewed_jsn_mask_dir,
                predicted_jsn_mask_dir,
                split,
                image_id,
            )
            if jsn_mask_path is not None:
                jsn_mask = load_jsn_mask(jsn_mask_path)
                if jsn_mask is None:
                    failure_writer.writerow({
                        "image_id": image_id,
                        "split": split,
                        "reason": f"{jsn_mode}_load_failed",
                    })
                else:
                    try:
                        rois = extract_jsn_guided_rois(enhanced, jsn_mask, roi_size=osteophyte_roi_size)
                        extraction_mode = str(jsn_mode)
                        reason = f"{jsn_mode}_success"
                    except Exception as exc:
                        failure_writer.writerow({
                            "image_id": image_id,
                            "split": split,
                            "reason": f"{jsn_mode}_failed:{type(exc).__name__}",
                        })

            if rois is None:
                try:
                    if args.geometry_only:
                        rois = extract_geometric_rois(enhanced, is_left, roi_size=osteophyte_roi_size)
                    elif args.skip_landmark_if_no_jsn:
                        rois = extract_geometric_rois(enhanced, is_left, roi_size=osteophyte_roi_size)
                        extraction_mode = "geometric_fallback"
                        reason = "no_jsn_mask_geometry_only"
                    else:
                        rois = extract_landmark_rois_with_timeout(
                            enhanced,
                            is_left,
                            backend=args.landmark_backend,
                            timeout_seconds=float(args.landmark_timeout_seconds),
                            roi_size=osteophyte_roi_size,
                        )
                        extraction_mode = "landmark"
                        reason = "landmark_success"
                except Exception as exc:
                    reason = f"landmark_failed:{type(exc).__name__}"
                    failure_writer.writerow({
                        "image_id": image_id,
                        "split": split,
                        "reason": reason,
                    })
                    rois = extract_geometric_rois(enhanced, is_left, roi_size=osteophyte_roi_size)
                    extraction_mode = "geometric_fallback"

            mode_counts[extraction_mode] += 1
            for site, roi in rois.items():
                cv2.imwrite(str(output_dir / split / f"{image_id}_{site}.png"), roi)
            audit_writer.writerow({
                "image_id": image_id,
                "split": split,
                "mode": extraction_mode,
                "backend": args.landmark_backend if not args.geometry_only else "none",
                "reason": reason,
            })
            processed += 1
            if index % progress_every == 0 or index == total_images:
                elapsed = time.monotonic() - started_at
                print(
                    f"[ROI] {index}/{total_images} images, "
                    f"jsn_reviewed={mode_counts['jsn_reviewed']}, "
                    f"jsn_predicted={mode_counts['jsn_predicted']}, "
                    f"landmark={mode_counts['landmark']}, "
                    f"fallback={mode_counts['geometric_fallback']}, "
                    f"skipped_existing={skipped_existing}, "
                    f"last={image_id}:{extraction_mode}, "
                    f"elapsed={elapsed:.1f}s",
                    flush=True,
                )
                audit_handle.flush()
                failure_handle.flush()

    failures_count = 0
    if failure_log_path.exists():
        with open(failure_log_path, "r", encoding="utf-8") as handle:
            failures_count = max(0, sum(1 for _ in handle) - 1)

    print(flush=True)
    print("=" * 70, flush=True)
    print("EXTRACTION COMPLETE", flush=True)
    print("=" * 70, flush=True)
    print(f"Processed: {processed} images", flush=True)
    print(f"Skipped existing: {skipped_existing} images", flush=True)
    print(f"JSN reviewed ROI: {mode_counts['jsn_reviewed']} images", flush=True)
    print(f"JSN predicted ROI: {mode_counts['jsn_predicted']} images", flush=True)
    print(f"Landmark ROI: {mode_counts['landmark']} images", flush=True)
    print(f"Geometric fallback: {mode_counts['geometric_fallback']} images", flush=True)
    print(f"Logged failures: {failures_count}", flush=True)
    if processed > 0:
        jsn_reviewed_rate = mode_counts["jsn_reviewed"] / processed * 100.0
        jsn_predicted_rate = mode_counts["jsn_predicted"] / processed * 100.0
        landmark_rate = mode_counts["landmark"] / processed * 100.0
        fallback_rate = mode_counts["geometric_fallback"] / processed * 100.0
        print(f"JSN reviewed rate: {jsn_reviewed_rate:.1f}%", flush=True)
        print(f"JSN predicted rate: {jsn_predicted_rate:.1f}%", flush=True)
        print(f"Landmark success rate: {landmark_rate:.1f}%", flush=True)
        print(f"Fallback rate: {fallback_rate:.1f}%", flush=True)

    total_rois = 0
    for split in ("train", "val", "test"):
        count = len(list((output_dir / split).glob("*.png")))
        print(f"  {split}: {count} ROIs", flush=True)
        total_rois += count
    print(f"Total ROIs saved: {total_rois}", flush=True)
    print(f"Audit log: {audit_log_path}", flush=True)
    print(f"Failure log: {failure_log_path}", flush=True)


if __name__ == "__main__":
    main()
