import gradio as gr
import cv2
import numpy as np
import pandas as pd
import torch
import time
import os
from pathlib import Path
from PIL import Image
from ultralytics import YOLO

# ──────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "best.pt"

CLASS_COLORS_BGR = {
    "missing_hole":    (59,  59,  255),
    "mouse_bite":      (83,  200,   0),
    "open_circuit":    (255, 121,  41),
    "short":           (0,   214, 255),
    "spur":            (249,   0, 213),
    "spurious_copper": (255, 229,   0),
}

HEX_COLORS = {
    "missing_hole":    "#FF3B3B",
    "mouse_bite":      "#00C853",
    "open_circuit":    "#2979FF",
    "short":           "#FFD600",
    "spur":            "#D500F9",
    "spurious_copper": "#00BCD4",
}

DEFECT_LABELS = {
    "missing_hole":    "Missing Hole",
    "mouse_bite":      "Mouse Bite",
    "open_circuit":    "Open Circuit",
    "short":           "Short Circuit",
    "spur":            "Spur",
    "spurious_copper": "Spurious Copper",
}

REMEDIES = {
    "missing_hole":    "Check the drill and re-drill the affected hole.",
    "mouse_bite":      "Check the board edge routing and replace worn blades.",
    "open_circuit":    "Re-solder the joint and test with a multimeter.",
    "short":           "Remove excess solder with solder wick and re-inspect.",
    "spur":            "Trim or etch the copper protrusion and re-inspect.",
    "spurious_copper": "Remove extra copper by etching and check mask alignment.",
}

# ──────────────────────────────────────────────
# MODEL LOADING
# ──────────────────────────────────────────────
_model  = None
_device = "cpu"

def load_model():
    global _model, _device
    if _model is not None:
        return _model, _device
    _device = "cuda:0" if torch.cuda.is_available() else "cpu"
    _model  = YOLO(str(MODEL_PATH))
    _model.to(_device)
    return _model, _device

try:
    load_model()
    MODEL_STATUS = "Ready"
    STATUS_OK    = True
except Exception as e:
    MODEL_STATUS = f"Error: {e}"
    STATUS_OK    = False

# ──────────────────────────────────────────────
# CORE HELPERS
# ──────────────────────────────────────────────
def get_severity(conf):
    if conf >= 0.80: return "High"
    if conf >= 0.50: return "Medium"
    return "Low"


def draw_boxes(img_bgr, detections):
    out = img_bgr.copy()
    for det in detections:
        cls  = det["defect_class"]
        conf = det["confidence"]
        x1, y1, x2, y2 = det["x1"], det["y1"], det["x2"], det["y2"]
        color = CLASS_COLORS_BGR.get(cls, (200, 200, 200))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 3)
        label = f"{DEFECT_LABELS.get(cls, cls)}  {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        by = max(y1 - 6, th + 8)
        cv2.rectangle(out, (x1, by - th - 8), (x1 + tw + 10, by + 2), color, -1)
        cv2.putText(out, label, (x1 + 5, by - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.LINE_AA)
    return out


def run_inference(pil_img, conf_thresh, iou_thresh, img_size, tta):
    model, device = load_model()
    img_bgr = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
    t0 = time.perf_counter()
    results = model.predict(
        source=img_bgr, conf=conf_thresh, iou=iou_thresh,
        imgsz=img_size, augment=tta, device=device, verbose=False,
    )
    elapsed_ms = (time.perf_counter() - t0) * 1000
    detections = []
    if results and results[0].boxes is not None:
        names = model.names
        for box in results[0].boxes:
            cls_name = names[int(box.cls[0])]
            conf     = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            detections.append({
                "defect_class": cls_name,
                "confidence":   round(conf, 4),
                "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                "width_px":  x2 - x1,
                "height_px": y2 - y1,
                "severity":  get_severity(conf),
                "remedy":    REMEDIES.get(cls_name, ""),
            })
    annotated_rgb = cv2.cvtColor(draw_boxes(img_bgr, detections), cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_rgb), detections, elapsed_ms


def build_df(detections, filename="image"):
    return pd.DataFrame([{
        "image_filename": filename,
        "defect_class":   d["defect_class"],
        "confidence":     d["confidence"],
        "x1": d["x1"], "y1": d["y1"], "x2": d["x2"], "y2": d["y2"],
        "width_px":   d["width_px"],
        "height_px":  d["height_px"],
        "severity":   d["severity"],
        "remedy":     d["remedy"],
    } for d in detections])


# ──────────────────────────────────────────────
# HTML BUILDERS
# ──────────────────────────────────────────────
def _dot(cls_name):
    c = HEX_COLORS.get(cls_name, "#999")
    return (f'<span style="display:inline-block;width:13px;height:13px;border-radius:50%;'
            f'background:{c};margin-right:8px;vertical-align:middle;'
            f'border:1.5px solid rgba(0,0,0,0.15);flex-shrink:0"></span>')


def _sev_badge(sev):
    s = {"High":   "background:#FFF0F0;color:#CC0000;border:2px solid #CC0000",
         "Medium": "background:#FFFBE6;color:#996600;border:2px solid #996600",
         "Low":    "background:#F0FFF4;color:#006600;border:2px solid #006600"
         }.get(sev, "background:#F5F5F5;color:#333;border:2px solid #ccc")
    return (f'<span style="{s};border-radius:20px;padding:2px 11px;'
            f'font-size:0.82rem;font-weight:700;white-space:nowrap">{sev}</span>')


def detection_html(detections, filename, elapsed_ms, w, h):
    count    = len(detections)
    is_clean = count == 0
    s_bg     = "#F0FFF4" if is_clean else "#FFF5F5"
    s_border = "#2E7D32" if is_clean else "#C53030"
    s_text   = "&#10004;  No Defects — Board looks clean!" if is_clean \
               else f"&#10008;  {count} Defect(s) Detected"

    h_str = f"""
    <div style="background:#fff;border:2px solid #E2E8F0;border-radius:14px;
                padding:20px 24px;margin-top:14px;font-family:Arial,sans-serif">
      <div style="display:flex;align-items:center;justify-content:space-between;
                  flex-wrap:wrap;gap:10px;margin-bottom:16px">
        <div>
          <div style="font-size:1rem;font-weight:800;color:#1A202C">{filename}</div>
          <div style="font-size:0.82rem;color:#718096;margin-top:3px">
            {w} x {h} px &nbsp;|&nbsp; Scan time: <b>{elapsed_ms:.0f} ms</b>
          </div>
        </div>
        <div style="background:{s_bg};border:2px solid {s_border};border-radius:10px;
                    padding:8px 16px;font-size:0.9rem;font-weight:700;color:{s_border}">
          {s_text}
        </div>
      </div>"""

    if detections:
        h_str += """
      <div style="overflow-x:auto">
        <table style="width:100%;border-collapse:collapse;font-size:0.88rem;color:#1A202C">
          <thead>
            <tr style="background:#F7FAFC;border-bottom:2px solid #E2E8F0">
              <th style="padding:10px 12px;text-align:left;color:#4A5568;font-weight:700">Defect Type</th>
              <th style="padding:10px 12px;text-align:left;color:#4A5568;font-weight:700">Confidence</th>
              <th style="padding:10px 12px;text-align:left;color:#4A5568;font-weight:700">Severity</th>
              <th style="padding:10px 12px;text-align:left;color:#4A5568;font-weight:700">Size (px)</th>
              <th style="padding:10px 12px;text-align:left;color:#4A5568;font-weight:700">What To Do</th>
            </tr>
          </thead>
          <tbody>"""
        for i, d in enumerate(detections):
            bg    = "#FAFAFA" if i % 2 == 0 else "#fff"
            pct   = int(d["confidence"] * 100)
            bar_c = "#CC0000" if pct >= 80 else "#996600" if pct >= 50 else "#006600"
            label = DEFECT_LABELS.get(d["defect_class"], d["defect_class"])
            h_str += f"""
            <tr style="background:{bg};border-bottom:1px solid #EDF2F7">
              <td style="padding:11px 12px">
                <div style="display:flex;align-items:center">
                  {_dot(d["defect_class"])}
                  <span style="font-weight:700;color:#2D3748">{label}</span>
                </div>
              </td>
              <td style="padding:11px 12px;min-width:105px">
                <div style="font-weight:700;color:#2D3748;margin-bottom:4px">{pct}%</div>
                <div style="background:#E2E8F0;border-radius:4px;height:6px;width:85px">
                  <div style="background:{bar_c};height:6px;border-radius:4px;
                              width:{min(pct,100)}%"></div>
                </div>
              </td>
              <td style="padding:11px 12px">{_sev_badge(d["severity"])}</td>
              <td style="padding:11px 12px;color:#4A5568">{d["width_px"]} x {d["height_px"]}</td>
              <td style="padding:11px 12px;color:#4A5568;max-width:250px;
                         line-height:1.5">{d["remedy"]}</td>
            </tr>"""
        h_str += "</tbody></table></div>"

    h_str += "</div>"
    return h_str


def batch_html(gallery_rows, all_df):
    total_imgs  = len(gallery_rows)
    total_defs  = len(all_df)
    clean       = sum(1 for *_, d, _, _, _, _ in gallery_rows if len(d) == 0)
    most_common = (DEFECT_LABELS.get(all_df["defect_class"].mode()[0], "—")
                   if not all_df.empty else "—")
    avg_conf    = f"{all_df['confidence'].mean()*100:.1f}%" if not all_df.empty else "—"

    cards = [
        ("Images Scanned", str(total_imgs), "#EBF8FF", "#1565C0"),
        ("Total Defects",  str(total_defs), "#FFF5F5", "#C53030"),
        ("Most Common",    most_common,     "#FFFAF0", "#C05621"),
        ("Avg Confidence", avg_conf,        "#F0FFF4", "#276749"),
        ("Clean Boards",   str(clean),      "#F0FFF4", "#276749"),
    ]
    cards_html = "".join(f"""
      <div style="background:{bg};border:2px solid {fg};border-radius:12px;
                  padding:14px 10px;text-align:center;flex:1;min-width:115px">
        <div style="font-size:1.45rem;font-weight:900;color:{fg};line-height:1;
                    margin-bottom:5px">{val}</div>
        <div style="font-size:0.73rem;color:#4A5568;font-weight:700;
                    text-transform:uppercase;letter-spacing:0.04em">{lbl}</div>
      </div>""" for lbl, val, bg, fg in cards)

    freq_rows = ""
    if not all_df.empty:
        freq = (all_df.groupby("defect_class")
                .agg(count=("defect_class","count"), avg_conf=("confidence","mean"))
                .reset_index().sort_values("count", ascending=False))
        for _, r in freq.iterrows():
            lbl = DEFECT_LABELS.get(r["defect_class"], r["defect_class"])
            freq_rows += f"""
            <tr style="border-bottom:1px solid #EDF2F7">
              <td style="padding:9px 12px">
                <div style="display:flex;align-items:center;font-weight:700;color:#2D3748">
                  {_dot(r["defect_class"])}{lbl}
                </div>
              </td>
              <td style="padding:9px 12px;font-weight:700;color:#2D3748">{int(r["count"])}</td>
              <td style="padding:9px 12px;color:#4A5568">{r["avg_conf"]*100:.1f}%</td>
            </tr>"""

    freq_section = ""
    if freq_rows:
        freq_section = f"""
        <div style="margin-top:18px">
          <div style="font-size:0.85rem;font-weight:800;color:#4A5568;margin-bottom:8px;
                      text-transform:uppercase;letter-spacing:0.05em">Defect Breakdown</div>
          <div style="overflow-x:auto">
            <table style="width:100%;border-collapse:collapse;font-size:0.88rem">
              <thead>
                <tr style="background:#F7FAFC;border-bottom:2px solid #E2E8F0">
                  <th style="padding:9px 12px;text-align:left;color:#4A5568;font-weight:700">Defect Type</th>
                  <th style="padding:9px 12px;text-align:left;color:#4A5568;font-weight:700">Count</th>
                  <th style="padding:9px 12px;text-align:left;color:#4A5568;font-weight:700">Avg Confidence</th>
                </tr>
              </thead>
              <tbody>{freq_rows}</tbody>
            </table>
          </div>
        </div>"""

    return f"""
    <div style="background:#fff;border:2px solid #E2E8F0;border-radius:14px;
                padding:22px;margin-top:22px;font-family:Arial,sans-serif">
      <div style="font-size:1.05rem;font-weight:900;color:#1A202C;margin-bottom:14px">
        Batch Summary — {total_imgs} Images Scanned
      </div>
      <div style="display:flex;gap:10px;flex-wrap:wrap">{cards_html}</div>
      {freq_section}
    </div>"""


def status_html():
    dev = _device.upper() if _device else "CPU"
    if STATUS_OK:
        return f"""
        <div style="font-family:Arial,sans-serif;padding:4px 0">
          <div style="display:inline-flex;align-items:center;gap:8px;
                      background:#E8F5E9;border:2px solid #2E7D32;border-radius:30px;
                      padding:6px 16px;font-size:0.9rem;font-weight:700;color:#2E7D32">
            &#10003; AI Model Ready &nbsp;|&nbsp; {dev}
          </div>
          <div style="margin-top:10px;font-size:0.85rem;color:#4A5568;line-height:1.8">
            <b style="color:#1A202C">Model:</b> Ghost+CBAM YOLOv8n<br>
            <b style="color:#1A202C">Size:</b> 6.3 MB &nbsp;|&nbsp;
            <b style="color:#1A202C">Parameters:</b> 3,006,818
          </div>
        </div>"""
    return f"""
        <div style="display:inline-flex;align-items:center;gap:8px;
                    background:#FFEBEE;border:2px solid #C62828;border-radius:30px;
                    padding:6px 16px;font-size:0.9rem;font-weight:700;color:#C62828">
          &#10007; Model Error — {MODEL_STATUS}
        </div>"""


def legend_html():
    items = "".join(
        f'<div style="display:flex;align-items:center;gap:10px;padding:5px 0;'
        f'font-size:0.9rem;font-weight:600;color:#2D3748;font-family:Arial,sans-serif">'
        f'<span style="width:15px;height:15px;border-radius:50%;background:{color};'
        f'display:inline-block;flex-shrink:0;border:2px solid rgba(0,0,0,0.15)"></span>'
        f'{DEFECT_LABELS[cls]}</div>'
        for cls, color in HEX_COLORS.items()
    )
    return f'<div>{items}</div>'


# ──────────────────────────────────────────────
# MAIN PROCESSING FUNCTION
# ──────────────────────────────────────────────
def process_images(files, conf_thresh, iou_thresh, img_size, tta):
    placeholder = ("<div style='padding:28px;text-align:center;color:#718096;"
                   "font-family:Arial,sans-serif;font-size:1rem;background:#fff;"
                   "border:2px solid #E2E8F0;border-radius:14px;margin-top:12px'>"
                   "Upload an image and press <b>SCAN NOW</b> to see results here.</div>")

    if files is None or len(files) == 0:
        return None, None, placeholder, None, "", None

    gallery_rows    = []
    all_dfs         = []
    all_tables_html = ""

    for f in files:
        pil_img = Image.open(f).convert("RGB")
        fname   = Path(f).name
        w, h    = pil_img.size
        annotated, detections, elapsed_ms = run_inference(
            pil_img, conf_thresh, iou_thresh, img_size, tta
        )
        gallery_rows.append((pil_img, annotated, detections, elapsed_ms, fname, w, h))
        df = build_df(detections, fname)
        if not df.empty:
            all_dfs.append(df)
        all_tables_html += detection_html(detections, fname, elapsed_ms, w, h)

    orig0, ann0, dets0, _, fname0, _, _ = gallery_rows[0]

    # Per-image CSV
    df0       = build_df(dets0, fname0)
    csv0_path = None
    if not df0.empty:
        csv0_path = "/tmp/result.csv"
        df0.to_csv(csv0_path, index=False)

    # Batch summary
    b_html    = ""
    batch_csv = None
    if len(gallery_rows) > 1:
        combined = pd.concat(all_dfs) if all_dfs else pd.DataFrame()
        b_html   = batch_html(gallery_rows, combined)
        if not combined.empty:
            batch_csv = "/tmp/batch_results.csv"
            combined.to_csv(batch_csv, index=False)

    return orig0, ann0, all_tables_html, csv0_path, b_html, batch_csv


# ──────────────────────────────────────────────
# CSS — passed to launch() for Gradio 6.x
# ──────────────────────────────────────────────
CSS = """
@import url('https://fonts.googleapis.com/css2?family=Nunito:wght@600;700;800;900&display=swap');

body, .gradio-container {
  background: #F0F4F8 !important;
  font-family: Arial, sans-serif !important;
  color: #1A202C !important;
}

/* White card panels */
.gradio-container .block {
  background: #ffffff !important;
  border: 2px solid #E2E8F0 !important;
  border-radius: 14px !important;
  box-shadow: 0 2px 8px rgba(0,0,0,0.05) !important;
}

/* All label text — dark and visible */
label span, .label-wrap span, span.svelte-1gfkn6j,
.block label > span:first-child {
  font-size: 0.95rem !important;
  font-weight: 700 !important;
  color: #1A202C !important;
}

/* Slider accent */
input[type=range] { accent-color: #1565C0 !important; }

/* Radio / checkbox */
input[type=radio]    { accent-color: #1565C0 !important; }
input[type=checkbox] { accent-color: #1565C0 !important;
                       width:18px; height:18px; }

/* Primary button — big blue SCAN NOW */
button.primary {
  background: #1565C0 !important;
  color: #ffffff !important;
  font-size: 1.1rem !important;
  font-weight: 900 !important;
  border-radius: 12px !important;
  border: none !important;
  padding: 14px 24px !important;
  box-shadow: 0 4px 14px rgba(21,101,192,0.3) !important;
  letter-spacing: 0.03em !important;
}
button.primary:hover { background: #0D47A1 !important; }

/* Secondary buttons */
button.secondary {
  background: #fff !important;
  color: #1565C0 !important;
  border: 2px solid #1565C0 !important;
  border-radius: 10px !important;
  font-weight: 700 !important;
}

/* File upload zone */
.upload-btn-wrap, .file-preview-holder {
  background: #E3F2FD !important;
  border: 3px dashed #90CAF9 !important;
  border-radius: 12px !important;
}

/* Hide Gradio footer */
footer { display: none !important; }
"""


# ──────────────────────────────────────────────
# GRADIO 6.x UI
# ──────────────────────────────────────────────
with gr.Blocks(title="PCB Defect Detection") as demo:

    # ── Top Banner ──
    gr.HTML("""
    <div style="background:linear-gradient(135deg,#1565C0,#1976D2);
                border-radius:16px;padding:22px 26px;margin-bottom:14px">
      <div style="font-size:1.55rem;font-weight:900;color:#ffffff;letter-spacing:0.01em">
        PCB Defect Detection
      </div>
      <div style="font-size:0.88rem;color:#BBDEFB;margin-top:5px">
        Upload a PCB board image — the AI will automatically find and highlight all defects
      </div>
    </div>
    """)

    with gr.Row():

        # ══════════════════════════════
        # LEFT COLUMN — Controls
        # ══════════════════════════════
        with gr.Column(scale=1, min_width=280):

            gr.HTML(status_html())

            gr.HTML("""<div style="font-size:1rem;font-weight:800;color:#1565C0;
                        margin:16px 0 6px 0;border-left:4px solid #1565C0;
                        padding-left:10px">Step 1 — Upload Image</div>""")

            file_input = gr.File(
                file_count="multiple",
                file_types=[".jpg", ".jpeg", ".png", ".bmp"],
                label="Click to upload  or  drag & drop your PCB image(s)",
            )

            gr.HTML("""<div style="font-size:1rem;font-weight:800;color:#1565C0;
                        margin:16px 0 6px 0;border-left:4px solid #1565C0;
                        padding-left:10px">Step 2 — Settings (Optional)</div>""")

            conf_slider = gr.Slider(
                minimum=0.05, maximum=0.95, value=0.15, step=0.05,
                label="Sensitivity  (lower = finds more defects)",
                info="Default 0.15 works well for most boards",
            )
            iou_slider = gr.Slider(
                minimum=0.20, maximum=0.90, value=0.45, step=0.05,
                label="Overlap Filter",
                info="Keep default unless you see duplicate boxes",
            )
            img_size_radio = gr.Radio(
                choices=[("Fast Scan (640px)", 640), ("High Accuracy (800px)", 800)],
                value=800,
                label="Scan Quality",
            )
            tta_check = gr.Checkbox(
                value=False,
                label="Extra Accuracy Mode  (slower scan)",
            )

            gr.HTML("""<div style="font-size:1rem;font-weight:800;color:#1565C0;
                        margin:16px 0 8px 0;border-left:4px solid #1565C0;
                        padding-left:10px">Step 3 — Run the Scan</div>""")

            run_btn = gr.Button("SCAN NOW", variant="primary", size="lg")

            gr.HTML("""<div style="font-size:0.82rem;font-weight:800;color:#4A5568;
                        margin:18px 0 6px 0;text-transform:uppercase;
                        letter-spacing:0.05em">Color Guide</div>""")
            gr.HTML(legend_html())

        # ══════════════════════════════
        # RIGHT COLUMN — Results
        # ══════════════════════════════
        with gr.Column(scale=3):

            gr.HTML("""<div style="font-size:1.1rem;font-weight:900;color:#1A202C;
                        margin-bottom:10px">Results</div>""")

            with gr.Row():
                orig_img = gr.Image(
                    label="Original Image",
                    type="pil",
                    interactive=False,
                    height=380,
                )
                det_img = gr.Image(
                    label="Detected Defects",
                    type="pil",
                    interactive=False,
                    height=380,
                )

            csv_dl = gr.File(
                label="Download Detection Results (CSV)",
                interactive=False,
            )

            table_out = gr.HTML(
                value=("<div style='padding:28px;text-align:center;color:#718096;"
                       "font-family:Arial,sans-serif;font-size:1rem;background:#fff;"
                       "border:2px solid #E2E8F0;border-radius:14px;margin-top:12px'>"
                       "Upload an image and press <b>SCAN NOW</b> to see results here.</div>")
            )

            batch_html_out = gr.HTML()

            batch_csv_dl = gr.File(
                label="Download Full Batch CSV",
                interactive=False,
            )

    # ── Wire up ──
    run_btn.click(
        fn=process_images,
        inputs=[file_input, conf_slider, iou_slider, img_size_radio, tta_check],
        outputs=[orig_img, det_img, table_out, csv_dl, batch_html_out, batch_csv_dl],
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        css=CSS,
    )
