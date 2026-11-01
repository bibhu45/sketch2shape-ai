"""CPU-only tests for the accuracy helpers: input detection, mask clean-up, orientation, fidelity."""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sketch2shape.metrics import fidelity_details  # noqa: E402
from sketch2shape.orient import ORIENTATION_PRESETS, apply_orientation, orientation_matrix  # noqa: E402
from sketch2shape.prepare import frame_object, refine_alpha  # noqa: E402
from sketch2shape.preprocess import detect_input_kind  # noqa: E402


def _line_mug():
    img = np.full((600, 800, 3), 255, np.uint8)
    cv2.rectangle(img, (250, 250), (550, 450), (30, 30, 30), 4)
    cv2.ellipse(img, (550, 350), (40, 70), 0, -90, 90, (30, 30, 30), 4)
    return Image.fromarray(img)


def _colour_picture():
    img = np.full((600, 800, 3), 255, np.uint8)
    cv2.ellipse(img, (400, 300), (220, 150), 0, 0, 360, (240, 160, 40), -1)
    cv2.circle(img, (330, 260), 40, (120, 70, 20), -1)
    return Image.fromarray(img)


def _grey_photo():
    rng = np.random.default_rng(1)
    base = np.linspace(60, 200, 800)[None, :].repeat(600, 0)
    noise = rng.normal(0, 12, (600, 800))
    g = np.clip(base + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(np.dstack([g, g, g]))


def test_detects_line_drawing_as_sketch():
    assert detect_input_kind(_line_mug()) == "sketch"


def test_detects_dark_canvas_sketch():
    inverted = Image.fromarray(255 - np.array(_line_mug()))
    assert detect_input_kind(inverted) == "sketch"


def test_detects_colour_image_as_picture():
    assert detect_input_kind(_colour_picture()) == "picture"


def test_detects_greyscale_photo_as_picture():
    assert detect_input_kind(_grey_photo()) == "picture"


def test_refine_alpha_drops_specks_keeps_big_parts():
    alpha = np.zeros((300, 300), np.uint8)
    alpha[50:200, 50:200] = 255          # main body
    alpha[220:260, 220:260] = 255        # separate but sizeable part (kept)
    alpha[5:8, 5:8] = 255                # speck (dropped)
    out = refine_alpha(alpha, min_part_share=0.05, erode_px=1)
    assert out[100, 100] == 255
    assert out[240, 240] == 255
    assert out[6, 6] == 0


def test_refine_alpha_removes_halo_edge():
    alpha = np.zeros((100, 100), np.uint8)
    alpha[20:80, 20:80] = 255
    out = refine_alpha(alpha, erode_px=1)
    assert out[20, 50] == 0 and out[22, 50] == 255


def test_frame_object_centres_and_pads():
    rgba = np.zeros((200, 100, 4), np.uint8)
    rgba[..., 3] = 255
    framed = np.array(frame_object(Image.fromarray(rgba), ratio=0.8, size=256))
    ys, xs = np.where(framed[..., 3] > 10)
    assert abs(((ys.min() + ys.max()) / 2) - 127.5) < 3
    assert abs(((xs.min() + xs.max()) / 2) - 127.5) < 3
    assert (ys.max() - ys.min()) <= 256 * 0.8 + 2


def test_orientation_presets_are_valid_rotations_or_mirrors():
    for preset in ORIENTATION_PRESETS:
        m = orientation_matrix(preset)
        assert np.allclose(m @ m.T, np.eye(3), atol=1e-9), preset   # orthogonal
        assert abs(abs(np.linalg.det(m)) - 1) < 1e-9, preset


def test_default_orientation_mirrors_so_faces_flip():
    verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float)
    faces = np.array([[0, 1, 2]])
    _, new_faces = apply_orientation(verts, faces, "Auto (recommended)")
    assert new_faces.tolist() == [[2, 1, 0]]
    _, raw_faces = apply_orientation(verts, faces, "Raw (no fix)")
    assert raw_faces.tolist() == [[0, 1, 2]]


def test_orientation_moves_triposr_axis_to_viewer_up():
    # The default fix must map exactly one axis onto +/-Y (it is a pure axis permutation).
    m = orientation_matrix("Auto (recommended)")
    assert sorted(np.abs(m).astype(int).sum(axis=0).tolist()) == [1, 1, 1]


def test_fidelity_rewards_matching_silhouette_and_punishes_podium():
    sketch = np.zeros((512, 512), np.uint8)
    cv2.rectangle(sketch, (150, 100), (360, 400), 255, 4)
    sketch_img = Image.fromarray(sketch)

    good = np.full((512, 512, 3), 255, np.uint8)
    cv2.rectangle(good, (150, 100), (360, 400), (200, 60, 60), -1)

    with_podium = good.copy()
    cv2.ellipse(with_podium, (255, 430), (230, 60), 0, 0, 360, (190, 190, 190), -1)

    unrelated = np.full((512, 512, 3), 255, np.uint8)
    cv2.circle(unrelated, (60, 60), 30, (0, 0, 0), -1)

    g = fidelity_details(sketch_img, Image.fromarray(good))
    p = fidelity_details(sketch_img, Image.fromarray(with_podium))
    u = fidelity_details(sketch_img, Image.fromarray(unrelated))
    assert g["score"] > 0.8
    assert p["score"] < g["score"] - 0.1     # the extra podium lowers the score
    assert u["score"] < 0.1


def test_open_outline_falls_back_to_edge_score():
    sketch = np.zeros((512, 512), np.uint8)
    cv2.line(sketch, (100, 100), (400, 100), 255, 4)   # one open line, nothing to fill
    gen = np.full((512, 512, 3), 255, np.uint8)
    cv2.line(gen, (100, 100), (400, 100), (0, 0, 0), 4)
    d = fidelity_details(Image.fromarray(sketch), Image.fromarray(gen))
    assert d["silhouette"] is None and d["score"] > 0.8
