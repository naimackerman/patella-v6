"""Gradio web interface for xrAI-OA pipeline."""

import io
from pathlib import Path

import gradio as gr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from app.inference import TriFQPipeline


def create_app(checkpoint_dir: str = "checkpoints") -> gr.Blocks:
    """Create the Gradio interface for xrAI-OA."""

    pipeline = TriFQPipeline(checkpoint_dir)

    def analyze_image(
        image_path: str,
        show_jsn: bool,
        show_osp: bool,
        show_scl: bool,
        kl_path: str,
    ):
        """Run full pipeline and return results."""
        if image_path is None:
            return None, "Please upload a knee X-ray image.", None

        results = pipeline.run(
            image_path,
            show_jsn=show_jsn,
            show_osteophytes=show_osp,
            show_sclerosis=show_scl,
            kl_path=kl_path,
        )

        # Convert overlay figure to image
        fig = results["overlay_figure"]
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
        buf.seek(0)
        plt.close(fig)

        # Grade distribution bar chart
        probs = results["kl_probabilities"]
        fig_dist, ax = plt.subplots(figsize=(6, 3))
        colors = ["#2ecc71", "#f39c12", "#e74c3c", "#9b59b6", "#1abc9c"]
        bars = ax.bar(["KL0", "KL1", "KL2", "KL3", "KL4"], probs, color=colors)
        ax.set_ylabel("Probability")
        ax.set_title(f"Predicted: KL {results['kl_grade']} ({results['kl_path_used']})")
        ax.set_ylim(0, 1)
        for bar, p in zip(bars, probs):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f"{p:.1%}", ha="center", fontsize=9)
        fig_dist.tight_layout()

        buf_dist = io.BytesIO()
        fig_dist.savefig(buf_dist, format="png", dpi=100)
        buf_dist.seek(0)
        plt.close(fig_dist)

        return buf, results["clinical_report"], buf_dist

    with gr.Blocks(title="xrAI-OA: Knee OA Assessment") as app:
        gr.Markdown("# xrAI-OA: Explainable Knee Osteoarthritis Assessment")
        gr.Markdown("Upload a knee X-ray for automated KL grade classification with tri-feature analysis.")

        with gr.Row():
            with gr.Column(scale=1):
                input_image = gr.Image(type="filepath", label="Upload Knee X-ray")
                with gr.Row():
                    show_jsn = gr.Checkbox(value=True, label="JSN Lines")
                    show_osp = gr.Checkbox(value=True, label="Osteophytes")
                    show_scl = gr.Checkbox(value=True, label="Sclerosis")
                kl_path = gr.Dropdown(
                    choices=["auto", "xgboost", "hybrid", "heuristic"],
                    value="auto",
                    label="KL Path",
                )
                analyze_btn = gr.Button("Analyze", variant="primary")

            with gr.Column(scale=1):
                output_image = gr.Image(label="Annotated X-ray")
                grade_chart = gr.Image(label="Grade Distribution")

        report_output = gr.Textbox(
            label="Clinical Report",
            lines=25,
            max_lines=40,
        )

        analyze_btn.click(
            fn=analyze_image,
            inputs=[input_image, show_jsn, show_osp, show_scl, kl_path],
            outputs=[output_image, report_output, grade_chart],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7860)
