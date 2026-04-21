"""Export trained models to ONNX format and benchmark inference time."""

import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.utils.device import get_device, clear_memory
from src.utils.seed import seed_everything


def export_jsn_segmenter(cfg: DictConfig, ckpt_dir: Path, export_dir: Path, device):
    """Export JSN segmentation model to ONNX."""
    from src.models.jsn_segmenter import create_jsn_segmenter

    ckpt_files = sorted((ckpt_dir / "jsn_segmenter").glob("*.ckpt"))
    if not ckpt_files:
        print("JSN segmenter checkpoint not found, skipping.")
        return None

    model = create_jsn_segmenter(cfg.model)
    checkpoint = torch.load(ckpt_files[-1], map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()
                  if k.startswith("model.")}
    model.load_state_dict(state_dict)
    model.eval()

    dummy_input = torch.randn(1, 1, 224, 224)
    onnx_path = export_dir / "jsn_segmenter.onnx"

    torch.onnx.export(
        model, dummy_input, str(onnx_path),
        input_names=["image"],
        output_names=["mask_logits"],
        dynamic_axes={"image": {0: "batch"}, "mask_logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported JSN segmenter to {onnx_path}")
    clear_memory()
    return onnx_path


def export_osteophyte_grader(cfg: DictConfig, ckpt_dir: Path, export_dir: Path, device):
    """Export osteophyte grading model to ONNX (single-site forward)."""
    from src.models.osteophyte_grader import OsteophyteGrader

    ckpt_files = sorted((ckpt_dir / "osteophyte").glob("*.ckpt"))
    if not ckpt_files:
        print("Osteophyte grader checkpoint not found, skipping.")
        return None

    model = OsteophyteGrader(cfg.model)
    checkpoint = torch.load(ckpt_files[-1], map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()
                  if k.startswith("model.")}
    model.load_state_dict(state_dict)
    model.eval()

    # Export with all 4 ROI inputs
    dummy_mf = torch.randn(1, 1, 140, 140)
    dummy_lf = torch.randn(1, 1, 140, 140)
    dummy_mt = torch.randn(1, 1, 140, 140)
    dummy_lt = torch.randn(1, 1, 140, 140)

    onnx_path = export_dir / "osteophyte_grader.onnx"

    # Wrap for ONNX export (dict output not supported)
    class OspWrapper(torch.nn.Module):
        def __init__(self, grader):
            super().__init__()
            self.grader = grader

        def forward(self, x_mf, x_lf, x_mt, x_lt):
            out = self.grader(x_mf, x_lf, x_mt, x_lt)
            return out["medial_femur"], out["lateral_femur"], out["medial_tibia"], out["lateral_tibia"]

    wrapper = OspWrapper(model)
    wrapper.eval()

    torch.onnx.export(
        wrapper, (dummy_mf, dummy_lf, dummy_mt, dummy_lt), str(onnx_path),
        input_names=["roi_mf", "roi_lf", "roi_mt", "roi_lt"],
        output_names=["grade_mf", "grade_lf", "grade_mt", "grade_lt"],
        opset_version=17,
    )
    print(f"Exported osteophyte grader to {onnx_path}")
    clear_memory()
    return onnx_path


def export_sclerosis_classifier(cfg: DictConfig, ckpt_dir: Path, export_dir: Path, device):
    """Export sclerosis classifier to ONNX."""
    from src.models.sclerosis_classifier import SclerosisClassifier

    ckpt_files = sorted((ckpt_dir / "sclerosis").glob("*.ckpt"))
    if not ckpt_files:
        print("Sclerosis classifier checkpoint not found, skipping.")
        return None

    model = SclerosisClassifier(cfg.model)
    checkpoint = torch.load(ckpt_files[-1], map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()
                  if k.startswith("model.")}
    model.load_state_dict(state_dict)
    model.eval()

    dummy_image = torch.randn(1, 1, 64, 64)
    dummy_texture = torch.randn(1, cfg.model.texture_feature_dim)
    dummy_side = torch.zeros(1, dtype=torch.long)
    onnx_path = export_dir / "sclerosis_classifier.onnx"

    torch.onnx.export(
        model, (dummy_image, dummy_texture, dummy_side), str(onnx_path),
        input_names=["roi_image", "texture_features", "side_ids"],
        output_names=["sclerosis_logits"],
        dynamic_axes={"roi_image": {0: "batch"}, "texture_features": {0: "batch"}, "side_ids": {0: "batch"},
                      "sclerosis_logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported sclerosis classifier to {onnx_path}")
    clear_memory()
    return onnx_path


def export_hybrid_classifier(cfg: DictConfig, ckpt_dir: Path, export_dir: Path, device):
    """Export hybrid KL classifier to ONNX."""
    from src.models.kl_hybrid import HybridKLClassifier

    ckpt_files = sorted((ckpt_dir / "kl_hybrid").glob("*.ckpt"))
    if not ckpt_files:
        print("Hybrid classifier checkpoint not found, skipping.")
        return None

    model = HybridKLClassifier(cfg.model)
    checkpoint = torch.load(ckpt_files[-1], map_location="cpu")
    state_dict = {k.replace("model.", ""): v for k, v in checkpoint["state_dict"].items()
                  if k.startswith("model.")}
    model.load_state_dict(state_dict)
    model.eval()

    dummy_image = torch.randn(1, 1, 224, 224)
    dummy_features = torch.randn(1, cfg.model.feature_dim)
    onnx_path = export_dir / "hybrid_kl_classifier.onnx"

    torch.onnx.export(
        model, (dummy_image, dummy_features), str(onnx_path),
        input_names=["xray_image", "feature_vector"],
        output_names=["kl_logits"],
        dynamic_axes={"xray_image": {0: "batch"}, "feature_vector": {0: "batch"},
                      "kl_logits": {0: "batch"}},
        opset_version=17,
    )
    print(f"Exported hybrid classifier to {onnx_path}")
    clear_memory()
    return onnx_path


def benchmark_onnx(onnx_path: Path, input_shapes: dict, n_runs: int = 50):
    """Benchmark ONNX model inference time."""
    try:
        import onnxruntime as ort
    except ImportError:
        print("onnxruntime not installed, skipping benchmark.")
        return None

    session = ort.InferenceSession(str(onnx_path))
    inputs = {name: np.random.randn(*shape).astype(np.float32)
              for name, shape in input_shapes.items()}

    # Warmup
    for _ in range(5):
        session.run(None, inputs)

    # Benchmark
    times = []
    for _ in range(n_runs):
        start = time.perf_counter()
        session.run(None, inputs)
        times.append(time.perf_counter() - start)

    avg_ms = np.mean(times) * 1000
    std_ms = np.std(times) * 1000
    print(f"  {onnx_path.name}: {avg_ms:.1f} +/- {std_ms:.1f} ms")
    return avg_ms


@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)
    device = get_device()

    ckpt_dir = Path(cfg.checkpoint_dir)
    export_dir = Path(cfg.output_dir) / "onnx"
    export_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("xrAI-OA Model Export to ONNX")
    print("=" * 60)

    # Export all models
    jsn_path = export_jsn_segmenter(cfg, ckpt_dir, export_dir, device)
    osp_path = export_osteophyte_grader(cfg, ckpt_dir, export_dir, device)
    scl_path = export_sclerosis_classifier(cfg, ckpt_dir, export_dir, device)
    hybrid_path = export_hybrid_classifier(cfg, ckpt_dir, export_dir, device)

    # Benchmark
    print("\n" + "=" * 60)
    print("ONNX Inference Benchmarks (CPU)")
    print("=" * 60)

    if jsn_path:
        benchmark_onnx(jsn_path, {"image": (1, 1, 224, 224)})
    if osp_path:
        benchmark_onnx(osp_path, {
            "roi_mf": (1, 1, 140, 140), "roi_lf": (1, 1, 140, 140),
            "roi_mt": (1, 1, 140, 140), "roi_lt": (1, 1, 140, 140),
        })
    if scl_path:
        benchmark_onnx(scl_path, {
            "roi_image": (1, 1, 64, 64),
            "texture_features": (1, cfg.model.texture_feature_dim),
        })
    if hybrid_path:
        benchmark_onnx(hybrid_path, {
            "xray_image": (1, 1, 224, 224),
            "feature_vector": (1, cfg.model.feature_dim),
        })

    print(f"\nONNX models saved to: {export_dir}")


if __name__ == "__main__":
    main()
