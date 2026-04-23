"""Gradio web interface for xrAI-OA pipeline."""

import io
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from app.inference import TriFQPipeline


CUSTOM_CSS = """
:root {
  --clinical-blue: #2563eb;
  --clinical-ink: #0f172a;
  --clinical-muted: #64748b;
}
.gradio-container {
  max-width: 1500px !important;
  margin: 0 auto !important;
  background: #f8fafc !important;
}
.clinical-hero {
  border: 1px solid #dbe3ef;
  background: linear-gradient(180deg, #ffffff 0%, #f1f5f9 100%);
  border-radius: 8px;
  padding: 18px 22px;
  margin-bottom: 14px;
}
.clinical-hero h1 {
  color: var(--clinical-ink);
  font-size: 25px;
  line-height: 1.2;
  margin: 0 0 6px 0;
  letter-spacing: 0;
}
.clinical-hero p {
  color: var(--clinical-muted);
  margin: 0;
  font-size: 14px;
}
.report-panel {
  border: 1px solid #dbe3ef;
  background: #ffffff;
  border-radius: 8px;
  padding: 14px 16px;
  color: #0f172a;
}
.report-panel h3 {
  margin-top: 0;
  margin-bottom: 12px;
  color: #0f172a;
  font-size: 16px;
}
.report-body {
  color: #1e293b;
  font-family: ui-monospace, SFMono-Regular, Menlo, Monaco, Consolas, "Liberation Mono", monospace;
  font-size: 13px;
  line-height: 1.55;
  white-space: normal;
}
#analyze-button button {
  min-height: 48px;
  font-size: 16px;
  font-weight: 700;
}
#annotated-image img,
#input-image img,
#grade-chart img {
  object-fit: contain !important;
}
"""


def _report_html(report: str) -> str:
    if not report:
        body = "Please upload and analyze a knee X-ray image."
    else:
        body = report
    escaped = (
        body.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace("\n", "<br>")
    )
    return f"""
<div class="report-panel">
  <h3>Structured Clinical Report</h3>
  <div class="report-body">{escaped}</div>
</div>
"""


def _figure_to_image(fig, dpi: int = 180, tight: bool = True) -> Image.Image:
    buf = io.BytesIO()
    if tight:
        fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    else:
        fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    plt.close(fig)
    return image


def _grade_chart(results: dict) -> Image.Image:
    probs = results["kl_probabilities"]
    labels = ["KL0", "KL1", "KL2", "KL3", "KL4"]
    pred_idx = int(results["kl_grade"])
    colors = ["#94a3b8"] * 5
    colors[pred_idx] = "#2563eb"
    fig_dist, ax = plt.subplots(figsize=(6.6, 3.0), facecolor="white")
    bars = ax.barh(labels, probs, color=colors, height=0.58)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Predicted probability", fontsize=9, color="#334155")
    ax.set_title(f"KL probability distribution ({results['kl_path_used']})", fontsize=11, color="#0f172a", pad=14)
    ax.invert_yaxis()
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis="both", colors="#475569", labelsize=9)
    for bar, p in zip(bars, probs):
        ax.text(
            min(float(p) + 0.015, 0.97),
            bar.get_y() + bar.get_height() / 2,
            f"{p:.1%}",
            va="center",
            fontsize=8.5,
            color="#0f172a",
            weight="bold" if p == max(probs) else "normal",
        )
    fig_dist.subplots_adjust(left=0.12, right=0.98, bottom=0.18, top=0.82)
    return _figure_to_image(fig_dist, dpi=100, tight=False)


def _cacheable_results(results: dict) -> dict:
    return {key: value for key, value in results.items() if key != "overlay_figure"}


def create_app(checkpoint_dir: str = "checkpoints") -> gr.Blocks:
    """Create the Gradio interface for xrAI-OA."""

    pipeline = TriFQPipeline(checkpoint_dir)
    kl_choices = ["auto", "hybrid", "heuristic"]
    if pipeline._xgboost_runtime_supported():
        kl_choices.insert(1, "xgboost")

    def analyze_image(
        image_path: str,
        show_jsn_medial: bool,
        show_jsn_lateral: bool,
        show_osp: bool,
        show_scl: bool,
        show_kl_badge: bool,
        display_preprocessing: str,
        distance_units: str,
        pixel_spacing_mm: float,
        kl_path: str,
    ):
        """Run full pipeline and return results."""
        if image_path is None:
            return None, _report_html("Please upload a knee X-ray image."), None, None

        results = pipeline.run(
            image_path,
            show_jsn=show_jsn_medial or show_jsn_lateral,
            show_jsn_medial=show_jsn_medial,
            show_jsn_lateral=show_jsn_lateral,
            show_osteophytes=show_osp,
            show_sclerosis=show_scl,
            show_kl_badge=show_kl_badge,
            display_preprocessing=display_preprocessing,
            distance_units=distance_units,
            pixel_spacing_mm=float(pixel_spacing_mm) if pixel_spacing_mm else None,
            kl_path=kl_path,
        )

        return (
            _figure_to_image(results["overlay_figure"]),
            _report_html(results["clinical_report"]),
            _grade_chart(results),
            _cacheable_results(results),
        )

    def refresh_display(
        image_path: str,
        cached_results: dict,
        show_jsn_medial: bool,
        show_jsn_lateral: bool,
        show_osp: bool,
        show_scl: bool,
        show_kl_badge: bool,
        display_preprocessing: str,
        distance_units: str,
        pixel_spacing_mm: float,
    ):
        """Redraw display-only controls without changing the assessment result."""
        if image_path is None or not cached_results:
            return None, _report_html("Please analyze a knee X-ray image first."), None, cached_results
        fig = pipeline.render_overlay_from_results(
            image_path,
            cached_results,
            show_jsn=show_jsn_medial or show_jsn_lateral,
            show_jsn_medial=show_jsn_medial,
            show_jsn_lateral=show_jsn_lateral,
            show_osteophytes=show_osp,
            show_sclerosis=show_scl,
            show_kl_badge=show_kl_badge,
            display_preprocessing=display_preprocessing,
            distance_units=distance_units,
            pixel_spacing_mm=float(pixel_spacing_mm) if pixel_spacing_mm else None,
        )
        return (
            _figure_to_image(fig),
            _report_html(cached_results["clinical_report"]),
            _grade_chart(cached_results),
            cached_results,
        )

    with gr.Blocks(
        title="xrAI-OA",
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
        css=CUSTOM_CSS,
    ) as app:
        gr.HTML(
            """
            <section class="clinical-hero">
              <h1>xrAI-OA</h1>
              <p>Explainable knee OA assessment with JSN measurement, osteophyte grading, sclerosis quantification, and KL prediction.</p>
            </section>
            """
        )

        with gr.Row():
            with gr.Column(scale=4, min_width=360):
                with gr.Group():
                    input_image = gr.Image(type="filepath", label="Input knee radiograph", height=300, elem_id="input-image")
                    kl_path = gr.Dropdown(
                        choices=kl_choices,
                        value="hybrid",
                        label="KL prediction path",
                    )
                    display_preprocessing = gr.Dropdown(
                        choices=[
                            ("Raw radiograph", "raw"),
                            ("CLAHE display", "clahe"),
                            ("Histogram-clipped display", "clip"),
                            ("Clip + CLAHE display", "clip_clahe"),
                        ],
                        value="raw",
                        label="Display preprocessing",
                    )
                    with gr.Row():
                        distance_units = gr.Dropdown(
                            choices=[
                                ("JSN distance in px", "px"),
                                ("JSN distance in mm", "mm"),
                                ("JSN distance in mm + px", "both"),
                            ],
                            value="px",
                            label="Overlay distance labels",
                        )
                        pixel_spacing_mm = gr.Number(
                            value=1.0,
                            label="Display scale mm/px",
                            precision=4,
                        )
                    with gr.Row():
                        show_jsn_medial = gr.Checkbox(value=True, label="Medial JSN")
                        show_jsn_lateral = gr.Checkbox(value=True, label="Lateral JSN")
                        show_osp = gr.Checkbox(value=True, label="Osteophytes")
                    with gr.Row():
                        show_scl = gr.Checkbox(value=True, label="Sclerosis")
                        show_kl_badge = gr.Checkbox(value=True, label="KL badge")
                    analyze_btn = gr.Button("Analyze radiograph", variant="primary", elem_id="analyze-button")

                grade_chart = gr.Image(label="KL probability distribution", height=260, elem_id="grade-chart")

            with gr.Column(scale=7, min_width=620):
                output_image = gr.Image(label="Annotated clinical overlay", height=620, elem_id="annotated-image")
                report_output = gr.HTML(label="Structured clinical report")

        analysis_inputs = [
            input_image,
            show_jsn_medial,
            show_jsn_lateral,
            show_osp,
            show_scl,
            show_kl_badge,
            display_preprocessing,
            distance_units,
            pixel_spacing_mm,
            kl_path,
        ]
        cached_results = gr.State()
        analysis_outputs = [output_image, report_output, grade_chart, cached_results]
        display_inputs = [
            input_image,
            cached_results,
            show_jsn_medial,
            show_jsn_lateral,
            show_osp,
            show_scl,
            show_kl_badge,
            display_preprocessing,
            distance_units,
            pixel_spacing_mm,
        ]

        analyze_btn.click(fn=analyze_image, inputs=analysis_inputs, outputs=analysis_outputs)
        for control in [
            show_jsn_medial,
            show_jsn_lateral,
            show_osp,
            show_scl,
            show_kl_badge,
            display_preprocessing,
            distance_units,
            pixel_spacing_mm,
        ]:
            control.change(fn=refresh_display, inputs=display_inputs, outputs=analysis_outputs)
        kl_path.change(fn=analyze_image, inputs=analysis_inputs, outputs=analysis_outputs)

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
