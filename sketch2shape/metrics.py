"""Timing, GPU memory and sketch-fidelity helpers."""
from __future__ import annotations

import time
from contextlib import contextmanager

import cv2
import numpy as np
from PIL import Image


class StageTimer:
    """Collects per-stage wall-clock time: `with timer.stage("generate"): ...`"""

    def __init__(self) -> None:
        self.times: dict[str, float] = {}

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.times[name] = round(self.times.get(name, 0.0) + time.perf_counter() - start, 2)

    @property
    def total(self) -> float:
        return round(sum(self.times.values()), 2)


def gpu_info() -> dict:
    """GPU name and peak memory so far. Safe to call without torch or CUDA."""
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return {
                "gpu": props.name,
                "vram_gb": round(props.total_memory / 1024**3, 1),
                "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            }
    except Exception:
        pass
    return {"gpu": None, "vram_gb": None, "peak_mem_gb": None}


def _fill_silhouette(mask: np.ndarray, close_px: int) -> np.ndarray:
    """Close small gaps in an outline, then fill the outside contour solid."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_px, close_px))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(closed)
    cv2.drawContours(out, contours, -1, 255, thickness=cv2.FILLED)
    return out


def fidelity_details(sketch_map: Image.Image, generated: Image.Image) -> dict:
    """How faithfully the generated image follows the sketch (0 to 1, higher is better).

    Two signals:
      edge       share of the sketch's line pixels that have a generated-image edge nearby
      silhouette overlap (IoU) between the sketch's filled outline and the generated object

    The silhouette term catches images that keep the lines but add a podium, a second
    object or a different overall shape. It only counts when the sketch outline is closed
    well enough to fill; otherwise the score falls back to the edge term alone.
    """
    gen_rgb = np.array(generated.convert("RGB"))
    sketch = np.array(sketch_map.convert("L").resize(generated.size))
    lines = (sketch > 127).astype(np.uint8) * 255
    line_area = int(np.count_nonzero(lines))
    if line_area == 0:
        return {"score": 0.0, "edge": 0.0, "silhouette": None}

    gen_gray = cv2.cvtColor(gen_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gen_gray, 60, 140)
    edges = cv2.dilate(edges, np.ones((7, 7), np.uint8), iterations=1)
    edge = float(np.logical_and(lines > 0, edges > 0).sum()) / float(line_area)

    sketch_fill = _fill_silhouette(lines, close_px=15)
    enclosed = np.count_nonzero(sketch_fill) > 2.5 * line_area
    silhouette = None
    if enclosed:
        not_white = (np.abs(gen_rgb.astype(np.int16) - 255).max(axis=2) > 28).astype(np.uint8) * 255
        gen_fill = _fill_silhouette(not_white, close_px=9)
        inter = np.count_nonzero(np.logical_and(sketch_fill > 0, gen_fill > 0))
        union = np.count_nonzero(np.logical_or(sketch_fill > 0, gen_fill > 0))
        silhouette = inter / union if union else 0.0

    score = edge if silhouette is None else 0.4 * edge + 0.6 * silhouette
    return {
        "score": round(float(score), 4),
        "edge": round(edge, 4),
        "silhouette": None if silhouette is None else round(float(silhouette), 4),
    }


def fidelity_score(sketch_map: Image.Image, generated: Image.Image) -> float:
    return fidelity_details(sketch_map, generated)["score"]
