"""Stage 4: background removal, mask clean-up, crop, centre and pad for reconstruction."""
from __future__ import annotations

import cv2
import numpy as np
from PIL import Image

_session = None


def _get_session():
    global _session
    if _session is None:
        from rembg import new_session

        # CPU on purpose: the ONNX GPU path fails with cuDNN errors on current Colab images,
        # and one 512 px image takes only a couple of seconds on CPU.
        _session = new_session("isnet-general-use", providers=["CPUExecutionProvider"])
    return _session


def refine_alpha(alpha: np.ndarray, min_part_share: float = 0.05, erode_px: int = 1) -> np.ndarray:
    """Clean a cut-out mask so it does not poison the 3D step.

    - hard threshold (no half-transparent fringe that bleeds the old background colour)
    - drop specks: keep parts that are at least `min_part_share` of the largest part
      (so a giraffe's head, body and legs survive, but dust does not)
    - shave `erode_px` off the edge to remove the white halo
    """
    mask = (alpha > 128).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if count > 1:
        areas = stats[1:, cv2.CC_STAT_AREA]
        biggest = int(areas.max())
        keep = np.zeros_like(mask)
        for index, area in enumerate(areas, start=1):
            if area >= min_part_share * biggest:
                keep[labels == index] = 1
        mask = keep
    if erode_px > 0:
        mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=int(erode_px))
    return mask * 255


def remove_background(img: Image.Image, min_part_share: float = 0.05, erode_px: int = 1) -> Image.Image:
    """Return a cleaned RGBA cut-out. Falls back to a white-background threshold if rembg fails."""
    rgba = None
    try:
        from rembg import remove

        out = np.array(remove(img.convert("RGB"), session=_get_session()).convert("RGBA"))
        if (out[..., 3] > 0).mean() > 0.01:
            rgba = out
    except Exception:
        pass
    if rgba is None:
        # Fallback: treat near-white pixels as background (we prompt for a white background).
        rgb = np.array(img.convert("RGB"))
        mask = (rgb.min(axis=2) < 235).astype(np.uint8) * 255
        rgba = np.dstack([rgb, mask])
    rgba = rgba.copy()
    rgba[..., 3] = refine_alpha(rgba[..., 3], min_part_share, erode_px)
    return Image.fromarray(rgba, "RGBA")


def frame_object(rgba: Image.Image, ratio: float = 0.85, size: int = 512) -> Image.Image:
    """Crop to the object, centre it, pad so it fills `ratio` of the frame."""
    arr = np.array(rgba)
    ys, xs = np.where(arr[..., 3] > 10)
    if len(ys) == 0:
        raise ValueError("No object found after background removal.")
    crop = Image.fromarray(arr[ys.min() : ys.max() + 1, xs.min() : xs.max() + 1])
    scale = (size * ratio) / max(crop.size)
    crop = crop.resize((max(1, int(crop.width * scale)), max(1, int(crop.height * scale))), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2), crop)
    return canvas


def prepare(
    img: Image.Image,
    ratio: float = 0.85,
    size: int = 512,
    min_part_share: float = 0.05,
    erode_px: int = 1,
) -> Image.Image:
    return frame_object(remove_background(img, min_part_share, erode_px), ratio=ratio, size=size)
