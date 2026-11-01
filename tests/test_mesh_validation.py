"""Needs trimesh (installed by requirements.txt). Skipped automatically if missing."""
import sys
from pathlib import Path

import pytest

trimesh = pytest.importorskip("trimesh")
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sketch2shape.config import load_config  # noqa: E402
from sketch2shape.mesh import clean_mesh, export_and_verify, validate_mesh  # noqa: E402


def _coloured_sphere():
    m = trimesh.creation.icosphere(subdivisions=3)
    rng = np.random.default_rng(0)
    m.visual.vertex_colors = rng.integers(0, 255, (len(m.vertices), 4), dtype=np.uint8)
    return m


def test_good_mesh_passes_and_round_trips(tmp_path):
    cfg = load_config()
    mesh = clean_mesh(_coloured_sphere())
    assert abs(float(mesh.extents.max()) - 1.0) < 1e-6
    report = validate_mesh(mesh, cfg)
    assert report["passed"], report
    export = export_and_verify(mesh, tmp_path, cfg)
    assert export["reload_ok"]
    assert (tmp_path / "model.glb").exists() and (tmp_path / "model.obj").exists()


def test_flat_plane_is_rejected():
    cfg = load_config()
    plane = trimesh.creation.box(extents=(1.0, 1.0, 0.001))
    assert validate_mesh(plane, cfg)["checks"]["has_volume"] is False
