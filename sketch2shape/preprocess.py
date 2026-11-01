"""Sketch preprocessing for Sketch2Shape AI.

Turns any sketch (pencil on paper, dark canvas, transparent PNG, Gradio canvas
output) into a clean white-lines-on-black square map, which is the input a
scribble ControlNet expects. Runs on CPU, no GPU needed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MAX_FILE_MB = 10
MIN_SIDE = 128
MAX_SIDE = 4096


class SketchError(ValueError):
    """Raised with a user-friendly message when the input cannot be used."""


def load_sketch(src: Union[str, Path, Image.Image, np.ndarray]) -> Image.Image:
    """Load and validate a sketch. Transparent areas are composited on white."""
    if isinstance(src, (str, Path)):
        path = Path(src)
        if path.suffix.lower() not in ALLOWED_EXTS:
            raise SketchError(
                f"Unsupported file type '{path.suffix}'. Use: {', '.join(sorted(ALLOWED_EXTS))}."
            )
        if not path.is_file():
            raise SketchError(f"File not found: {path}")
        if path.stat().st_size > MAX_FILE_MB * 1024 * 1024:
            raise SketchError(f"File is larger than {MAX_FILE_MB} MB.")
        try:
            img = Image.open(path)
            img.load()
        except Exception as exc:  # unreadable or corrupt file
            raise SketchError("The file could not be read as an image.") from exc
    elif isinstance(src, np.ndarray):
        img = Image.fromarray(src)
    elif isinstance(src, Image.Image):
        img = src
    else:
        raise SketchError("Unsupported input type.")

    if img.mode in ("RGBA", "LA") or (img.mode == "P" and "transparency" in img.info):
        rgba = img.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        img = Image.alpha_composite(background, rgba)
    img = img.convert("RGB")

    w, h = img.size
    if min(w, h) < MIN_SIDE:
        raise SketchError(f"Image is too small. Use at least {MIN_SIDE} px on each side.")
    if max(w, h) > MAX_SIDE:
        scale = MAX_SIDE / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def detect_input_kind(img: Image.Image) -> str:
    """Return 'sketch' for line drawings or 'picture' for photos and colour images.

    A line drawing is mostly paper with a thin layer of dark, colourless ink. A picture
    has colour, or a large share of mid-tone pixels. The user can always override this.
    """
    small = np.asarray(img.convert("RGB").resize((256, 256), Image.BILINEAR)).astype(np.float32) / 255.0
    gray = small.mean(axis=2)
    if np.median(gray) < 0.45:  # dark canvas: flip so the paper is light
        small, gray = 1.0 - small, 1.0 - gray
    paper = max(float(np.percentile(gray, 95)), 0.3)
    g = np.clip(gray / paper, 0.0, 1.0)
    marked = g < 0.85
    share = float(marked.mean())
    if share < 0.002:
        return "sketch"
    mx, mn = small.max(axis=2), small.min(axis=2)
    saturation = (mx - mn) / np.maximum(mx, 1e-6)
    if float(saturation[marked].mean()) > 0.22 or share > 0.30:
        return "picture"
    return "sketch"


def prepare_picture(src: Union[str, Path, Image.Image, np.ndarray], max_side: int = 1024) -> Image.Image:
    """Validate a photo or colour image and return it as RGB, capped in size."""
    img = load_sketch(src)
    w, h = img.size
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    return img


def preprocess(
    src: Union[str, Path, Image.Image, np.ndarray],
    size: int = 512,
    margin: float = 0.08,
    min_ink_ratio: float = 0.0015,
) -> tuple[Image.Image, dict]:
    """Return (scribble_map, info).

    scribble_map: RGB PIL image, white lines on black, square, size x size.
    info: dict of simple measurements for logging and metadata.json.
    """
    img = load_sketch(src)
    gray = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2GRAY)

    # Work at a moderate resolution so thresholds behave consistently.
    h, w = gray.shape
    work_max = 1024
    if max(h, w) > work_max:
        scale = work_max / max(h, w)
        gray = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    # Light denoise and local contrast boost (helps faint pencil and phone photos).
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)

    # Make the background light. If the image is mostly dark, invert it.
    inverted = bool(np.median(gray) < 127)
    if inverted:
        gray = 255 - gray

    # Ink = pixels darker than their local neighbourhood. Copes with uneven lighting.
    ink = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 12
    )

    # Remove tiny specks of noise.
    count, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    min_area = max(8, int(0.00002 * ink.size))
    cleaned = np.zeros_like(ink)
    for i in range(1, count):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            cleaned[labels == i] = 255
    ink = cleaned

    ink_ratio = float(np.count_nonzero(ink)) / ink.size
    if ink_ratio < min_ink_ratio:
        raise SketchError(
            "The sketch looks blank or too faint. Draw with a darker, thicker line and try again."
        )

    # Crop to the drawing, add a margin, pad to a square.
    ys, xs = np.where(ink > 0)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    crop = ink[y0:y1, x0:x1]
    ch, cw = crop.shape
    side = int(max(ch, cw) * (1 + 2 * margin))
    canvas = np.zeros((side, side), dtype=np.uint8)
    oy, ox = (side - ch) // 2, (side - cw) // 2
    canvas[oy : oy + ch, ox : ox + cw] = crop

    # Resize, thicken lines slightly so the ControlNet can see them, re-binarise.
    canvas = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    kernel = np.ones((3, 3), np.uint8)
    canvas = cv2.dilate(canvas, kernel, iterations=1)
    _, canvas = cv2.threshold(canvas, 40, 255, cv2.THRESH_BINARY)

    info = {
        "input_size": list(img.size),
        "output_size": [size, size],
        "inverted": inverted,
        "ink_ratio": round(ink_ratio, 5),
    }
    return Image.fromarray(canvas).convert("RGB"), info


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Preprocess a sketch into a scribble map.")
    parser.add_argument("input")
    parser.add_argument("--out", default="sketch_processed.png")
    parser.add_argument("--size", type=int, default=512)
    args = parser.parse_args()
    try:
        scribble, details = preprocess(args.input, size=args.size)
    except SketchError as err:
        raise SystemExit(f"Input problem: {err}")
    scribble.save(args.out)
    print(f"Saved {args.out}  {details}")
