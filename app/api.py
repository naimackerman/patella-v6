"""FastAPI wrapper around the KOA-TriFQ inference pipeline."""

from __future__ import annotations

import base64
import io
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from fastapi import FastAPI, File, Form, HTTPException, UploadFile

from app.inference import TriFQPipeline


app = FastAPI(title="KOA-TriFQ API", version="1.0.0")
pipeline = TriFQPipeline("checkpoints")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    include_overlay: bool = Form(False),
    kl_path: str = Form("auto"),
):
    suffix = Path(file.filename or "upload.png").suffix or ".png"
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(await file.read())
            tmp_path = Path(tmp.name)
        results = pipeline.run(str(tmp_path), kl_path=kl_path)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        if "tmp_path" in locals() and tmp_path.exists():
            tmp_path.unlink(missing_ok=True)

    payload = {
        "kl_grade": results["kl_grade"],
        "kl_confidence": results["kl_confidence"],
        "kl_probabilities": results["kl_probabilities"],
        "kl_path_used": results["kl_path_used"],
        "kl_predictions": results["kl_predictions"],
        "jsn_features": results["jsn_features"],
        "osp_features": results["osp_features"],
        "scl_features": results["scl_features"],
        "clinical_report": results["clinical_report"],
    }
    if include_overlay and results.get("overlay_figure") is not None:
        buf = io.BytesIO()
        results["overlay_figure"].savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        payload["overlay_png_base64"] = base64.b64encode(buf.read()).decode("ascii")
        plt.close(results["overlay_figure"])
    return payload
