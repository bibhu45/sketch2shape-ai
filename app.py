"""Sketch2Shape AI web app: draw a sketch or upload a picture, get a textured 3D model.

Run: python app.py            (local)
     python app.py --share    (Colab: prints a public link)
"""
import argparse
import queue
import tempfile
import threading
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image

from sketch2shape.config import load_config
from sketch2shape.orient import ORIENTATION_PRESETS
from sketch2shape.pipeline import Pipeline, PipelineError

cfg = load_config()
pipe = Pipeline(cfg)

MODES = {
    "Auto-detect": "auto",
    "Sketch (line drawing)": "sketch",
    "Picture (photo or colour image)": "picture",
}
QUALITIES = list(cfg.get("quality_presets", {"Balanced": {}}).keys())
STYLE_CHIPS = {
    "Ceramic": "glossy white ceramic",
    "Wood": "polished natural wood",
    "Metal": "brushed metal",
    "Plush": "soft plush fabric toy",
    "Realistic": "photorealistic, natural colours",
    "Cartoon": "stylised cartoon, bold colours",
}
STAGES = ["Input", "Redraw", "Cut-out", "3D model", "Checks"]

CSS = """
<style>
.gradio-container { max-width: 1280px !important; margin: auto; }
.s2s-hero { padding: 28px 30px; border-radius: 18px; margin-bottom: 14px; color: #fff;
  background: linear-gradient(120deg, #4338ca 0%, #7c3aed 55%, #db2777 100%); }
.s2s-hero h1 { margin: 0 0 6px; font-size: 2rem; letter-spacing: -0.02em; color: #fff; }
.s2s-hero p { margin: 0; opacity: .92; font-size: 1.02rem; color: #fff; }
.s2s-steps { display: flex; gap: 10px; flex-wrap: wrap; margin-top: 16px; }
.s2s-steps span { background: rgba(255,255,255,.18); border-radius: 999px; padding: 5px 14px; font-size: .88rem; }
.s2s-status { display: flex; gap: 6px; flex-wrap: wrap; margin: 4px 0 8px; }
.s2s-pill { padding: 5px 12px; border-radius: 999px; font-size: .84rem; border: 1px solid #c7c7d1; color: #6b7280; }
.s2s-pill.done { background: #dcfce7; border-color: #86efac; color: #166534; }
.s2s-pill.now { background: #ede9fe; border-color: #a78bfa; color: #5b21b6; font-weight: 600; }
.s2s-pill.fail { background: #fee2e2; border-color: #fca5a5; color: #991b1b; }
.s2s-msg { margin: 2px 0 10px; font-size: .95rem; opacity: .85; }
.s2s-tip { font-size: .85rem; opacity: .75; }
#generate-btn { font-size: 1.05rem; min-height: 48px; }
@media (prefers-color-scheme: dark) {
  .s2s-pill { border-color: #3f3f46; color: #a1a1aa; }
  .s2s-pill.done { background: #052e16; border-color: #166534; color: #86efac; }
  .s2s-pill.now { background: #2e1065; border-color: #6d28d9; color: #ddd6fe; }
  .s2s-pill.fail { background: #450a0a; border-color: #991b1b; color: #fecaca; }
}
</style>
"""

HERO = """
<div class="s2s-hero">
  <h1>Sketch2Shape AI</h1>
  <p>Turn a hand-drawn sketch or any picture into a textured 3D model you can download as .glb or .obj.</p>
  <div class="s2s-steps"><span>1 · Draw or upload</span><span>2 · Add a hint (optional)</span><span>3 · Generate</span><span>4 · Rotate and download</span></div>
</div>
"""


def status_html(stage: int, message: str = "", failed: bool = False) -> str:
    """stage = index of the stage in progress; STAGES past it are pending, before it done."""
    pills = []
    for i, name in enumerate(STAGES):
        if failed and i == stage:
            cls = "fail"
        elif i < stage:
            cls = "done"
        elif i == stage:
            cls = "now"
        else:
            cls = ""
        mark = "✓ " if cls == "done" else ""
        pills.append(f'<span class="s2s-pill {cls}">{mark}{name}</span>')
    msg = f'<div class="s2s-msg">{message}</div>' if message else ""
    return f'<div class="s2s-status">{"".join(pills)}</div>{msg}'


def _to_pil(value):
    """Normalise whatever a Gradio image/sketch component returns into an RGB PIL image on white."""
    if value is None:
        return None
    if isinstance(value, dict):  # ImageEditor / Sketchpad format
        value = value.get("composite")
        if value is None:
            return None
    if isinstance(value, (str, Path)):
        value = Image.open(value)
    if isinstance(value, np.ndarray):
        value = Image.fromarray(value.astype("uint8"))
    if value.mode in ("RGBA", "LA", "P"):
        value = value.convert("RGBA")
        bg = Image.new("RGBA", value.size, (255, 255, 255, 255))
        value = Image.alpha_composite(bg, value)
    return value.convert("RGB")


def _blank(img: Image.Image) -> bool:
    return img is None or np.asarray(img.convert("L")).min() > 245


def _report_md(r: dict) -> str:
    v = r["validation"]
    checks = " · ".join(f"{'✅' if ok else '❌'} {name.replace('_', ' ')}" for name, ok in v["checks"].items())
    fid = "n/a (picture input)" if r["fidelity"] is None else f"{r['fidelity']:.2f}  (tries: {', '.join(f'{s:.2f}' for s in r['candidate_scores'])})"
    times = " · ".join(f"{k} {t:.1f}s" for k, t in r["timings_sec"].items())
    hw = r["hardware"]
    gpu = (hw.get("gpu") or hw.get("device") or "unknown") if isinstance(hw, dict) else str(hw)
    return (
        f"### {'Model ready' if r['success'] else 'Model built, but a check failed'}\n"
        f"**Input treated as:** {r['mode_used']}  ·  **Quality:** {r['quality']}  ·  **Seed:** {r['seed']}\n\n"
        f"**Match to your sketch:** {fid}\n\n"
        f"**Mesh:** {v['faces']:,} faces · {v['vertices']:,} vertices · bounding box {v['bbox_extents']}\n\n"
        f"**Checks:** {checks}\n\n"
        f"**Time:** {r['total_sec']:.1f}s total ({times})  ·  **Hardware:** {gpu}\n\n"
        "If the model looks sideways or upside down, open *Advanced* and change **Model orientation**."
    )


def generate(drawn, uploaded, prompt, mode_label, quality, strength, seed, orientation, smooth):
    img = _to_pil(uploaded) if uploaded is not None else _to_pil(drawn)
    if _blank(img):
        raise gr.Error("Draw something on the canvas or upload an image first.")

    events: "queue.Queue" = queue.Queue()
    holder = {}

    def work():
        try:
            holder["result"] = pipe.run(
                img,
                prompt=(prompt or "").strip() or None,
                seed=int(seed),
                sketch_strength=float(strength),
                mode=MODES.get(mode_label, "auto"),
                quality=quality,
                orientation=orientation,
                smooth=bool(smooth),
                progress=lambda m: events.put(("msg", m)),
                on_image=lambda k, im: events.put(("img", (k, im))),
            )
        except PipelineError as err:
            holder["error"] = str(err)
        except Exception as err:  # never leave the UI hanging
            holder["error"] = f"Unexpected error: {err}"
        finally:
            events.put(("end", None))

    threading.Thread(target=work, daemon=True).start()

    shown = {"input": None, "reference": None, "cutout": None}
    stage, message = 0, "Starting"
    stage_of = {"Checking": 0, "Drawing": 1, "Removing": 2, "Building": 3, "Cleaning": 4}
    # outputs: status, input, reference, cutout, viewer, files, report
    yield status_html(0, message), None, None, None, None, None, ""
    while True:
        try:
            kind, payload = events.get(timeout=0.5)
        except queue.Empty:
            continue
        if kind == "end":
            break
        if kind == "msg":
            message = payload
            for prefix, idx in stage_of.items():
                if payload.startswith(prefix) or payload.startswith("Result drifted"):
                    stage = 1 if payload.startswith("Result drifted") else idx
        else:
            key, image = payload
            shown[key] = image
        yield status_html(stage, message), shown["input"], shown["reference"], shown["cutout"], None, None, ""

    if "error" in holder:
        yield (
            status_html(stage, "", failed=True) + f'<div class="s2s-msg">⚠️ {holder["error"]}</div>',
            shown["input"], shown["reference"], shown["cutout"], None, None, "",
        )
        return
    r = holder["result"]
    yield (
        status_html(len(STAGES), "Done. Drag to rotate the model, scroll to zoom."),
        shown["input"], shown["reference"], shown["cutout"],
        r["glb"], [r["glb"], r["obj"]], _report_md(r),
    )


def clear():
    return None, None, "", status_html(-1), None, None, None, None, None, ""


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Sketch2Shape AI") as demo:
        gr.HTML(CSS + HERO)
        with gr.Row(equal_height=False):
            with gr.Column(scale=5):
                with gr.Tabs():
                    with gr.Tab("✏️ Draw"):
                        try:
                            drawn = gr.Sketchpad(
                                type="pil", image_mode="RGB", label="Draw one object, large, with a clear outline",
                                brush=gr.Brush(default_size=4, colors=["#000000"], color_mode="fixed"),
                                canvas_size=(640, 520), height=560,
                            )
                        except Exception:
                            drawn = gr.Sketchpad(label="Draw one object, large, with a clear outline")
                        gr.HTML('<div class="s2s-tip">Tip: draw one object, fill most of the canvas, close the outline.</div>')
                    with gr.Tab("🖼️ Upload"):
                        uploaded = gr.Image(type="pil", label="Upload a sketch, photo or picture (png, jpg, webp)", height=560)
                        gr.HTML('<div class="s2s-tip">Pictures with the object on a plain background work best.</div>')
                mode = gr.Radio(list(MODES), value="Auto-detect", label="Input type",
                                info="Auto-detect tells a line drawing from a picture. Pictures skip the redraw step for higher accuracy.")
                prompt = gr.Textbox(label="Describe it (optional)", lines=2,
                                    placeholder="e.g. a red ceramic coffee mug  ·  leave empty to let the sketch speak")
                with gr.Row():
                    chips = [gr.Button(name, size="sm") for name in STYLE_CHIPS]
                quality = gr.Radio(QUALITIES, value="Balanced" if "Balanced" in QUALITIES else QUALITIES[0], label="Quality",
                                   info="Fast ≈ 30 s · Balanced tries 3 versions and keeps the best · Best is finest mesh")
                with gr.Accordion("Advanced", open=False):
                    strength = gr.Slider(0.5, 2.0, value=cfg["generation"]["sketch_strength"], step=0.05,
                                         label="Sketch strength", info="Higher follows your lines more strictly")
                    seed = gr.Number(value=cfg["seed"], precision=0, label="Seed", info="Same seed + same sketch = same result")
                    orientation = gr.Dropdown(ORIENTATION_PRESETS, value=ORIENTATION_PRESETS[0], label="Model orientation",
                                              info="Change this if the model appears sideways or upside down")
                    smooth = gr.Checkbox(value=True, label="Smooth the surface")
                with gr.Row():
                    go = gr.Button("✨ Generate 3D model", variant="primary", elem_id="generate-btn", scale=3)
                    reset = gr.Button("Clear", scale=1)
            with gr.Column(scale=6):
                status = gr.HTML(status_html(-1))
                viewer = gr.Model3D(label="3D model (drag to rotate, scroll to zoom)", height=520)
                with gr.Row():
                    img_in = gr.Image(label="What we read", height=200, interactive=False)
                    img_ref = gr.Image(label="Redrawn / reference", height=200, interactive=False)
                    img_cut = gr.Image(label="Cut-out sent to 3D", height=200, interactive=False)
                files = gr.File(label="Download (.glb opens in most viewers; .obj for Blender etc.)", file_count="multiple")
                report = gr.Markdown()

        for chip, text in zip(chips, STYLE_CHIPS.values()):
            chip.click(lambda p, t=text: (f"{p.rstrip(', ')}, {t}" if p else t), inputs=prompt, outputs=prompt)

        go.click(
            generate,
            inputs=[drawn, uploaded, prompt, mode, quality, strength, seed, orientation, smooth],
            outputs=[status, img_in, img_ref, img_cut, viewer, files, report],
        )
        reset.click(clear, outputs=[drawn, uploaded, prompt, status, img_in, img_ref, img_cut, viewer, files, report])

        sample = Path("samples/inputs/mug.png")
        if sample.exists():
            try:
                gr.Examples(examples=[[str(sample), "a white ceramic mug"]], inputs=[uploaded, prompt], label="Try an example")
            except Exception:
                pass
    return demo


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", action="store_true", help="create a public link (use in Colab)")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()
    demo = build_ui()
    try:
        demo.queue(default_concurrency_limit=1)
    except TypeError:
        demo.queue(concurrency_count=1)
    demo.launch(share=args.share, server_port=args.port)
