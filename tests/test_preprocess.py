import sys
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sketch2shape.preprocess import SketchError, preprocess  # noqa: E402


def _mug(dark=False):
    img = np.full((600, 800, 3), 255, np.uint8)
    cv2.rectangle(img, (250, 250), (550, 450), (30, 30, 30), 4)
    cv2.ellipse(img, (550, 350), (40, 70), 0, -90, 90, (30, 30, 30), 4)
    return Image.fromarray(255 - img if dark else img)


def test_output_is_white_lines_on_black_square():
    out, info = preprocess(_mug(), size=512)
    arr = np.array(out)
    assert out.size == (512, 512)
    assert arr.mean() < 100          # mostly black
    assert arr.max() == 255          # has white lines


def test_dark_background_is_inverted():
    _, info = preprocess(_mug(dark=True))
    assert info["inverted"] is True


def test_blank_sketch_rejected():
    with pytest.raises(SketchError):
        preprocess(Image.new("RGB", (400, 400), "white"))


def test_tiny_image_rejected():
    with pytest.raises(SketchError):
        preprocess(Image.new("RGB", (50, 50), "white"))


def test_transparent_png_handled():
    rgba = Image.new("RGBA", (600, 600), (0, 0, 0, 0))
    arr = np.array(rgba)
    cv2.rectangle(arr, (150, 150), (450, 450), (0, 0, 0, 255), 5)
    out, _ = preprocess(Image.fromarray(arr))
    assert np.array(out).max() == 255
