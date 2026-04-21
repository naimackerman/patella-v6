"""Generate a PDF clinical report for one image using the inference pipeline."""

from __future__ import annotations

import io
from pathlib import Path

import hydra
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig

from app.inference import TriFQPipeline
from src.xai.pdf_export import save_clinical_report_pdf
from src.utils.seed import seed_everything


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    image_path = Path(cfg.get("image_path", ""))
    if not image_path.exists():
        raise FileNotFoundError("Set image_path=/absolute/path/to/image.png when calling scripts/export_pdf_report.py")

    output_pdf = Path(cfg.get("output_pdf", Path(cfg.result_dir) / f"{image_path.stem}_report.pdf"))
    pipeline = TriFQPipeline(cfg.checkpoint_dir)
    results = pipeline.run(str(image_path))

    overlay_bytes = None
    if results.get("overlay_figure") is not None:
        buf = io.BytesIO()
        results["overlay_figure"].savefig(buf, format="png", dpi=160, bbox_inches="tight")
        buf.seek(0)
        overlay_bytes = buf.getvalue()
        plt.close(results["overlay_figure"])

    save_clinical_report_pdf(
        output_path=output_pdf,
        title=f"KOA-TriFQ Clinical Report: {image_path.name}",
        report_text=results["clinical_report"],
        overlay_png_bytes=overlay_bytes,
    )
    print(f"Saved PDF report to {output_pdf}")


if __name__ == "__main__":
    main()
