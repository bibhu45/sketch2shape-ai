"""Mesh orientation presets (pure NumPy, no trimesh needed).

TripoSR builds meshes in its own axis convention, so a raw result shows up lying on
its side in a standard Y-up viewer (glTF, Blender, Gradio). The default preset applies
the same correction TripoSR's own demo uses: rotate -90 deg about X, mirror Z, rotate
+90 deg about Y. The other presets are extra turns on top of it, so a wrong-looking
model can be fixed live from the UI without touching code.
"""
from __future__ import annotations

import numpy as np

ORIENTATION_PRESETS = [
    "Auto (recommended)",
    "Raw (no fix)",
    "Flip upside down",
    "Rotate X +90",
    "Rotate X -90",
    "Rotate Y +90",
    "Rotate Z +90",
]


def _rot(axis: str, degrees: float) -> np.ndarray:
    t = np.radians(degrees)
    c, s = np.cos(t), np.sin(t)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def _base() -> np.ndarray:
    # Same order as TripoSR's demo: Rx(-90) first, then mirror Z, then Ry(+90).
    return _rot("y", 90) @ np.diag([1.0, 1.0, -1.0]) @ _rot("x", -90)


def orientation_matrix(preset: str = "Auto (recommended)") -> np.ndarray:
    """3x3 matrix applied to vertices as `v' = M @ v`."""
    if preset == "Raw (no fix)":
        return np.eye(3)
    base = _base()
    extra = {
        "Auto (recommended)": np.eye(3),
        "Flip upside down": _rot("x", 180),
        "Rotate X +90": _rot("x", 90),
        "Rotate X -90": _rot("x", -90),
        "Rotate Y +90": _rot("y", 90),
        "Rotate Z +90": _rot("z", 90),
    }.get(preset, np.eye(3))
    return extra @ base


def apply_orientation(vertices: np.ndarray, faces: np.ndarray, preset: str):
    """Return (vertices, faces) rotated. Face winding is flipped when the matrix mirrors."""
    m = orientation_matrix(preset)
    verts = np.asarray(vertices, dtype=float) @ m.T
    faces = np.asarray(faces)
    if np.linalg.det(m) < 0:
        faces = faces[:, ::-1]
    return verts, faces
