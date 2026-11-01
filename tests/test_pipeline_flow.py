"""Full pipeline flow on CPU with fake models. Needs trimesh; skipped if missing.

The diffusion model and TripoSR are replaced by tiny fakes, so this checks everything
else end to end: mode detection, candidate selection, cut-out, orientation, validation,
export and the metadata written to disk. Run it on your laptop, no GPU needed.
"""
import json
import sys
from pathlib import Path

import pytest

trimesh = pytest.importorskip("trimesh")
import cv2  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sketch2shape import prepare as prepare_module  # noqa: E402
from sketch2shape.config import load_config  # noqa: E402
from sketch2shape.pipeline import Pipeline, PipelineError  # noqa: E402


class FakeGenerator:
    """Returns the sketch's filled outline in colour, so fidelity is high and varies by seed."""

    def __init__(self):
        self.calls = []

    def generate(self, scribble, prompt, seed=None, strength=None, steps=None):
        self.calls.append((seed, strength))
        lines = np.array(scribble.convert("L")) > 127
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        closed = cv2.morphologyEx(lines.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        img = np.full((512, 512, 3), 255, np.uint8)
        cv2.drawContours(img, cnts, -1, (200 - (seed or 0) % 5 * 5, 90, 60), cv2.FILLED)
        return Image.fromarray(img)


class FakeReconstructor:
    def __init__(self):
        self.resolutions = []

    def reconstruct(self, rgba, mc_resolution=None):
        self.resolutions.append(mc_resolution)
        mesh = trimesh.creation.icosphere(subdivisions=4)
        mesh.apply_scale([1.0, 0.7, 0.9])
        rng = np.random.default_rng(0)
        mesh.visual.vertex_colors = rng.integers(0, 255, (len(mesh.vertices), 4), dtype=np.uint8)
        return mesh


@pytest.fixture
def pipe(monkeypatch):
    cfg = load_config()
    cfg["mesh"]["min_faces"] = 100
    p = Pipeline(cfg)
    p._gen, p._rec = FakeGenerator(), FakeReconstructor()
    # No rembg in the test: the white-background fallback inside remove_background is used.
    monkeypatch.setattr(prepare_module, "_get_session", lambda: (_ for _ in ()).throw(RuntimeError("no rembg")))
    return p


def _line_mug(path):
    img = np.full((640, 640, 3), 255, np.uint8)
    pts = np.array([[200, 230], [440, 230], [420, 470], [220, 470]], np.int32)
    cv2.polylines(img, [pts], True, (35, 35, 35), 5)
    Image.fromarray(img).save(path)
    return path


def _colour_picture(path):
    img = np.full((640, 640, 3), 255, np.uint8)
    cv2.ellipse(img, (320, 320), (200, 140), 0, 0, 360, (240, 160, 40), -1)
    Image.fromarray(img).save(path)
    return path


def test_sketch_mode_runs_candidates_and_exports(pipe, tmp_path):
    events = []
    result = pipe.run(
        _line_mug(tmp_path / "mug.png"), prompt="a mug", out_dir=tmp_path / "run",
        quality="Balanced", progress=events.append, on_image=lambda k, im: events.append(k),
    )
    assert result["mode_used"] == "sketch"
    assert len(result["candidate_scores"]) >= 3           # Balanced = 3 candidates
    assert result["fidelity"] == max(result["candidate_scores"])
    assert pipe._rec.resolutions == [256]                 # Balanced mesh resolution
    assert result["success"], result["validation"]
    assert {"input", "reference", "cutout"} <= set(e for e in events if isinstance(e, str) and e in ("input", "reference", "cutout"))
    for name in ("model.glb", "model.obj", "metadata.json", "generated.png", "cutout.png"):
        assert (tmp_path / "run" / name).exists(), name
    meta = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert meta["mode_used"] == "sketch"


def test_picture_mode_skips_diffusion(pipe, tmp_path):
    result = pipe.run(_colour_picture(tmp_path / "pic.png"), out_dir=tmp_path / "run", quality="Best")
    assert result["mode_used"] == "picture" and result["auto_detected"] == "picture"
    assert result["fidelity"] is None
    assert pipe._gen.calls == []                          # the picture went straight to 3D
    assert pipe._rec.resolutions == [320]                 # Best mesh resolution
    assert result["success"], result["validation"]


def test_mode_can_be_forced(pipe, tmp_path):
    result = pipe.run(_colour_picture(tmp_path / "pic.png"), out_dir=tmp_path / "run", mode="sketch", quality="Fast")
    assert result["mode_used"] == "sketch" and result["auto_detected"] == "picture"


def test_blank_input_gives_friendly_error(pipe, tmp_path):
    Image.new("RGB", (400, 400), "white").save(tmp_path / "blank.png")
    with pytest.raises(PipelineError):
        pipe.run(tmp_path / "blank.png", out_dir=tmp_path / "run", mode="sketch")
