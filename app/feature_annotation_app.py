"""Gradio app for osteophyte and sclerosis review-sheet annotation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from tempfile import gettempdir

import cv2
import gradio as gr
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont


CSV_PATH = Path(
    "/Users/naim/Desktop/patella-v4/annotations/packages/feature_grading/feature_review_template.csv"
)

OSTEO_FIELDS = [
    ("final_osp_mf", "MF", "suggestion_osp_mf", "confidence_mf"),
    ("final_osp_lf", "LF", "suggestion_osp_lf", "confidence_lf"),
    ("final_osp_mt", "MT", "suggestion_osp_mt", "confidence_mt"),
    ("final_osp_lt", "LT", "suggestion_osp_lt", "confidence_lt"),
]
SCL_FIELDS = [
    ("final_scl_medial", "Sclerosis Medial", "suggestion_scl_medial", "scl_confidence_med"),
    ("final_scl_lateral", "Sclerosis Lateral", "suggestion_scl_lateral", "scl_confidence_lat"),
]
OSTEO_CHOICES = ["", "0", "1", "2", "3"]
SCL_CHOICES = ["", "0", "1", "2"]
CONF_CHOICES = ["", "low", "medium", "high"]
DISPLAY_MODE_CHOICES = ["Raw", "CLAHE", "CLAHE+Clip"]
ANNOTATION_IMAGE_DIR = Path(gettempdir()) / "koa_trifq_annotation_app"
ANNOTATION_IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def load_sheet(csv_path: str) -> pd.DataFrame:
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Feature review CSV not found: {path}")
    return pd.read_csv(path).fillna("")


def save_sheet(df: pd.DataFrame, csv_path: str):
    df.to_csv(csv_path, index=False)


def row_completion_status(row: pd.Series) -> str:
    required = [field for field, *_ in OSTEO_FIELDS + SCL_FIELDS]
    filled = sum(str(row.get(field, "")).strip() != "" for field in required)
    return f"{filled}/{len(required)} completed"


def normalize_choice(value, choices) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text == "" or text.lower() == "nan":
        return ""
    try:
        numeric = float(text)
        if numeric.is_integer():
            text = str(int(numeric))
    except ValueError:
        pass
    return text if text in choices else ""


def sanitize_grade(value, allow_blank: bool = True):
    value = str(value).strip()
    if value == "" and allow_blank:
        return ""
    return int(float(value))


def infer_knee_side(image_id: str) -> str:
    name = str(image_id).strip().upper()
    if name.endswith("L"):
        return "Left"
    if name.endswith("R"):
        return "Right"
    return "Unknown"


def _histogram_clip(image: np.ndarray, low_pct: int = 5, high_pct: int = 99) -> np.ndarray:
    low = np.percentile(image, low_pct)
    high = np.percentile(image, high_pct)
    if high - low < 1e-6:
        return image.astype(np.uint8)
    clipped = np.clip(image, low, high)
    return ((clipped - low) / (high - low) * 255).astype(np.uint8)


def _apply_display_preprocessing(image: np.ndarray, display_mode: str) -> np.ndarray:
    mode = display_mode.strip() if display_mode else "Raw"
    if mode == "Raw":
        return image

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    proc = clahe.apply(image)
    if mode == "CLAHE+Clip":
        proc = _histogram_clip(proc, low_pct=5, high_pct=99)
    return proc


def render_tagged_image(image_path: str | None, image_id: str, display_mode: str = "Raw") -> str | None:
    if not image_path:
        return None

    source = Path(image_path)
    if not source.exists():
        return None

    base = cv2.imread(str(source), cv2.IMREAD_GRAYSCALE)
    if base is None:
        return None
    proc = _apply_display_preprocessing(base, display_mode)
    image = Image.fromarray(proc).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    width, height = image.size
    knee_side = infer_knee_side(image_id)
    if knee_side == "Right":
        corner_map = {
            "LF": "top_left",
            "MF": "top_right",
            "LT": "bottom_left",
            "MT": "bottom_right",
        }
    else:
        corner_map = {
            "MF": "top_left",
            "LF": "top_right",
            "MT": "bottom_left",
            "LT": "bottom_right",
        }
    colors = {
        "MF": "#cf3f2b",
        "LF": "#2864c7",
        "MT": "#2d9b53",
        "LT": "#b27419",
    }

    margin = 8

    for tag, corner in corner_map.items():
        text_box = draw.textbbox((0, 0), tag, font=font)
        text_w = text_box[2] - text_box[0]

        if corner == "top_left":
            x0, y0 = margin, margin
        elif corner == "top_right":
            x0, y0 = width - margin - text_w, margin
        elif corner == "bottom_left":
            x0, y0 = margin, height - margin - (text_box[3] - text_box[1])
        else:
            x0, y0 = width - margin - text_w, height - margin - (text_box[3] - text_box[1])

        draw.text(
            (x0, y0),
            tag,
            fill=colors[tag],
            font=font,
            stroke_width=1,
            stroke_fill="white",
        )

    safe_mode = display_mode.lower().replace("+", "_plus_").replace(" ", "_")
    out_path = ANNOTATION_IMAGE_DIR / f"{source.stem}_{safe_mode}_tagged.png"
    image.save(out_path)
    return str(out_path)


def get_row_payload(df: pd.DataFrame, index: int, display_mode: str = "Raw"):
    row = df.iloc[index]
    source_image_path = row.get("local_image_path") or row.get("path") or None
    image_path = render_tagged_image(source_image_path, row["image_id"], display_mode=display_mode)
    status = (
        f"Index {index + 1}/{len(df)} | "
        f"Image `{row['image_id']}` | Split `{row['split']}` | KL `{row['kl_grade']}` | "
        f"{row_completion_status(row)}"
    )
    image_meta = (
        f"Filename: `{row['image_id']}` | Side: `{infer_knee_side(row['image_id'])}` | "
        f"Display: `{display_mode}`"
    )

    suggestions = [
        f"Image ID: {row['image_id']}",
        f"Split: {row['split']}",
        f"KL grade: {row['kl_grade']}",
        "",
        "Osteophyte suggestions:",
    ]
    for final_col, label, suggestion_col, conf_col in OSTEO_FIELDS:
        suggestions.append(f"{label}: suggested {row.get(suggestion_col, '')}")
    suggestions.append("")
    suggestions.append("Sclerosis suggestions:")
    for final_col, label, suggestion_col, conf_col in SCL_FIELDS:
        suggestions.append(f"{label}: suggested {row.get(suggestion_col, '')}")
    suggestion_text = "\n".join(suggestions)

    outputs = [
        image_path,
        status,
        image_meta,
        suggestion_text,
        str(row.get("notes", "")),
    ]
    for final_col, _label, _suggestion_col, conf_col in OSTEO_FIELDS:
        outputs.append(normalize_choice(row.get(final_col, ""), OSTEO_CHOICES))
        outputs.append(normalize_choice(row.get(conf_col, ""), CONF_CHOICES))
    for final_col, _label, _suggestion_col, conf_col in SCL_FIELDS:
        outputs.append(normalize_choice(row.get(final_col, ""), SCL_CHOICES))
        outputs.append(normalize_choice(row.get(conf_col, ""), CONF_CHOICES))
    outputs.append(index)
    return outputs


def save_current(
    csv_path: str,
    index: int,
    display_mode: str,
    notes: str,
    *field_values,
):
    df = load_sheet(csv_path)
    row_index = int(index)

    cursor = 0
    for final_col, _label, _suggestion_col, conf_col in OSTEO_FIELDS:
        df.at[row_index, final_col] = sanitize_grade(field_values[cursor], allow_blank=True)
        df.at[row_index, conf_col] = str(field_values[cursor + 1]).strip()
        cursor += 2
    for final_col, _label, _suggestion_col, conf_col in SCL_FIELDS:
        df.at[row_index, final_col] = sanitize_grade(field_values[cursor], allow_blank=True)
        df.at[row_index, conf_col] = str(field_values[cursor + 1]).strip()
        cursor += 2

    df.at[row_index, "notes"] = str(notes).strip()
    save_sheet(df, csv_path)
    return get_row_payload(df, row_index, display_mode=display_mode)


def save_and_move(
    direction: int,
    csv_path: str,
    index: int,
    display_mode: str,
    notes: str,
    *field_values,
):
    outputs = save_current(csv_path, index, display_mode, notes, *field_values)
    df = load_sheet(csv_path)
    new_index = min(max(int(index) + direction, 0), len(df) - 1)
    return get_row_payload(df, new_index, display_mode=display_mode)


def jump_to_index(csv_path: str, target_index: int, display_mode: str):
    df = load_sheet(csv_path)
    idx = min(max(int(target_index), 1), len(df)) - 1
    return get_row_payload(df, idx, display_mode=display_mode)


def jump_to_incomplete(csv_path: str, display_mode: str):
    df = load_sheet(csv_path)
    required = [field for field, *_ in OSTEO_FIELDS + SCL_FIELDS]
    incomplete = df.index[
        ~df[required].astype(str).apply(lambda col: col.str.strip() != "").all(axis=1)
    ]
    idx = int(incomplete[0]) if len(incomplete) > 0 else 0
    return get_row_payload(df, idx, display_mode=display_mode)


def refresh_current_row(csv_path: str, index: int, display_mode: str):
    df = load_sheet(csv_path)
    return get_row_payload(df, int(index), display_mode=display_mode)


def refresh_display_only(csv_path: str, index: int, display_mode: str):
    df = load_sheet(csv_path)
    row = df.iloc[int(index)]
    source_image_path = row.get("local_image_path") or row.get("path") or None
    image_path = render_tagged_image(source_image_path, row["image_id"], display_mode=display_mode)
    image_meta = (
        f"Filename: `{row['image_id']}` | Side: `{infer_knee_side(row['image_id'])}` | "
        f"Display: `{display_mode}`"
    )
    return image_path, image_meta


def reset_current_row(csv_path: str, index: int, display_mode: str):
    df = load_sheet(csv_path)
    row_index = int(index)
    for final_col, _label, _suggestion_col, conf_col in OSTEO_FIELDS:
        df.at[row_index, final_col] = ""
        df.at[row_index, conf_col] = ""
    for final_col, _label, _suggestion_col, conf_col in SCL_FIELDS:
        df.at[row_index, final_col] = ""
        df.at[row_index, conf_col] = ""
    df.at[row_index, "notes"] = ""
    save_sheet(df, csv_path)
    return get_row_payload(df, row_index, display_mode=display_mode)


def reset_all_manual_entries(csv_path: str, index: int, display_mode: str):
    df = load_sheet(csv_path)
    csv_file = Path(csv_path)
    backup_path = csv_file.with_name(
        f"{csv_file.stem}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{csv_file.suffix}"
    )
    df.to_csv(backup_path, index=False)

    for final_col, _label, _suggestion_col, conf_col in OSTEO_FIELDS:
        df[final_col] = ""
        df[conf_col] = ""
    for final_col, _label, _suggestion_col, conf_col in SCL_FIELDS:
        df[final_col] = ""
        df[conf_col] = ""
    df["notes"] = ""

    save_sheet(df, csv_path)
    outputs = get_row_payload(df, int(index), display_mode=display_mode)
    outputs[1] = f"{outputs[1]}\n\nReset all manual entries. Backup saved to `{backup_path}`."
    return outputs


def apply_suggestions(csv_path: str, index: int, notes: str, *field_values):
    df = load_sheet(csv_path)
    row = df.iloc[int(index)]
    outputs = [str(notes)]
    for final_col, _label, suggestion_col, conf_col in OSTEO_FIELDS:
        current_grade = normalize_choice(
            row.get(final_col, "") if str(row.get(final_col, "")).strip() else row.get(suggestion_col, ""),
            OSTEO_CHOICES,
        )
        current_conf = normalize_choice(row.get(conf_col, "") or "medium", CONF_CHOICES)
        outputs.extend([current_grade, current_conf])
    for final_col, _label, suggestion_col, conf_col in SCL_FIELDS:
        current_grade = normalize_choice(
            row.get(final_col, "") if str(row.get(final_col, "")).strip() else row.get(suggestion_col, ""),
            SCL_CHOICES,
        )
        current_conf = normalize_choice(row.get(conf_col, "") or "medium", CONF_CHOICES)
        outputs.extend([current_grade, current_conf])
    return outputs


def create_app(csv_path: str = str(CSV_PATH)) -> gr.Blocks:
    df = load_sheet(csv_path)
    initial_mode = "Raw"
    initial = get_row_payload(df, 0, display_mode=initial_mode)

    css = """
    .gradio-container {max-width: 100% !important;}
    .full-height {min-height: 85vh;}
    .panel-title {margin-bottom: 0.4rem;}
    .image-panel img {object-fit: contain !important; width: 100% !important; height: 100% !important;}
    """

    with gr.Blocks(title="KOA-TriFQ Feature Annotation", css=css, fill_height=True) as app:
        gr.Markdown("# KOA-TriFQ Feature Annotation")

        csv_state = gr.State(csv_path)
        row_index_state = gr.State(initial[-1])

        status_box = gr.Markdown(value=initial[1])

        with gr.Column(elem_classes=["full-height"]):
            with gr.Row(equal_height=True):
                with gr.Column(scale=7):
                    display_mode = gr.Radio(
                        choices=DISPLAY_MODE_CHOICES,
                        value=initial_mode,
                        label="Display Mode",
                    )
                    image_view = gr.Image(
                        value=initial[0],
                        type="filepath",
                        label="Image",
                        height=520,
                        elem_classes=["image-panel"],
                    )
                    image_meta_box = gr.Markdown(value=initial[2])
                with gr.Column(scale=5):
                    suggestion_box = gr.Textbox(
                        value=initial[3],
                        label="Suggestions",
                        lines=24,
                        interactive=False,
                    )

            with gr.Row(equal_height=True):
                with gr.Column(scale=7):
                    gr.Markdown("## Osteophyte")
                    with gr.Row():
                        osp_mf = gr.Dropdown(choices=OSTEO_CHOICES, value=initial[5], label="MF")
                        osp_lf = gr.Dropdown(choices=OSTEO_CHOICES, value=initial[7], label="LF")
                        osp_mt = gr.Dropdown(choices=OSTEO_CHOICES, value=initial[9], label="MT")
                        osp_lt = gr.Dropdown(choices=OSTEO_CHOICES, value=initial[11], label="LT")
                    with gr.Row():
                        conf_mf = gr.Dropdown(choices=CONF_CHOICES, value=initial[6], label="MF Confidence")
                        conf_lf = gr.Dropdown(choices=CONF_CHOICES, value=initial[8], label="LF Confidence")
                        conf_mt = gr.Dropdown(choices=CONF_CHOICES, value=initial[10], label="MT Confidence")
                        conf_lt = gr.Dropdown(choices=CONF_CHOICES, value=initial[12], label="LT Confidence")

                with gr.Column(scale=5):
                    gr.Markdown("## Sclerosis")
                    with gr.Row():
                        scl_med = gr.Dropdown(choices=SCL_CHOICES, value=initial[13], label="Medial")
                        scl_lat = gr.Dropdown(choices=SCL_CHOICES, value=initial[15], label="Lateral")
                    with gr.Row():
                        scl_conf_med = gr.Dropdown(choices=CONF_CHOICES, value=initial[14], label="Medial Confidence")
                        scl_conf_lat = gr.Dropdown(choices=CONF_CHOICES, value=initial[16], label="Lateral Confidence")
                    notes = gr.Textbox(value=initial[4], label="Notes", lines=5)

        with gr.Row():
            prev_btn = gr.Button("Save + Previous")
            save_btn = gr.Button("Save")
            next_btn = gr.Button("Save + Next", variant="primary")
            suggest_btn = gr.Button("Use Suggestions")
            reset_btn = gr.Button("Reset Current")
            reset_all_btn = gr.Button("Reset All Manual Entries", variant="stop")
            jump_index = gr.Number(value=1, precision=0, label="Jump To Row")
            jump_btn = gr.Button("Open Row")
            incomplete_btn = gr.Button("First Incomplete")

        common_inputs = [
            csv_state,
            row_index_state,
            display_mode,
            notes,
            osp_mf, conf_mf,
            osp_lf, conf_lf,
            osp_mt, conf_mt,
            osp_lt, conf_lt,
            scl_med, scl_conf_med,
            scl_lat, scl_conf_lat,
        ]
        common_outputs = [
            image_view,
            status_box,
            image_meta_box,
            suggestion_box,
            notes,
            osp_mf, conf_mf,
            osp_lf, conf_lf,
            osp_mt, conf_mt,
            osp_lt, conf_lt,
            scl_med, scl_conf_med,
            scl_lat, scl_conf_lat,
            row_index_state,
        ]

        save_btn.click(save_current, inputs=common_inputs, outputs=common_outputs)
        prev_btn.click(lambda *args: save_and_move(-1, *args), inputs=common_inputs, outputs=common_outputs)
        next_btn.click(lambda *args: save_and_move(1, *args), inputs=common_inputs, outputs=common_outputs)
        suggest_btn.click(
            apply_suggestions,
            inputs=[csv_state, row_index_state, notes, osp_mf, conf_mf, osp_lf, conf_lf, osp_mt, conf_mt, osp_lt, conf_lt, scl_med, scl_conf_med, scl_lat, scl_conf_lat],
            outputs=[notes, osp_mf, conf_mf, osp_lf, conf_lf, osp_mt, conf_mt, osp_lt, conf_lt, scl_med, scl_conf_med, scl_lat, scl_conf_lat],
        )
        reset_btn.click(
            reset_current_row,
            inputs=[csv_state, row_index_state, display_mode],
            outputs=common_outputs,
        )
        reset_all_btn.click(
            reset_all_manual_entries,
            inputs=[csv_state, row_index_state, display_mode],
            outputs=common_outputs,
        )
        jump_btn.click(jump_to_index, inputs=[csv_state, jump_index, display_mode], outputs=common_outputs)
        incomplete_btn.click(jump_to_incomplete, inputs=[csv_state, display_mode], outputs=common_outputs)
        display_mode.change(
            refresh_display_only,
            inputs=[csv_state, row_index_state, display_mode],
            outputs=[image_view, image_meta_box],
        )

    return app


if __name__ == "__main__":
    app = create_app()
    app.launch(server_name="0.0.0.0", server_port=7861, share=True)
