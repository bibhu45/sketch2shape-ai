"""Stage 5: image-to-3D reconstruction.

TripoSR is the safe baseline (light, installs easily, fits a free Colab T4).
InstantMesh is the planned upgrade; add it as a second class with the same
`.reconstruct(rgba) -> trimesh.Trimesh` interface and switch with config.yaml.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent


def _ensure_marching_cubes() -> None:
    """TripoSR imports `torchmcubes`, which must be compiled and often fails on Colab.

    If it is not installed, register a drop-in replacement built on scikit-image
    (CPU marching cubes, same call signature and return types).
    """
    try:
        import torchmcubes  # noqa: F401

        return
    except Exception:
        pass

    import types

    import torch
    from skimage import measure

    def marching_cubes(vol, thresh):
        verts, faces, _, _ = measure.marching_cubes(vol.detach().cpu().numpy(), level=float(thresh))
        return (
            torch.from_numpy(np.ascontiguousarray(verts)).float(),
            torch.from_numpy(np.ascontiguousarray(faces).astype(np.int64)),
        )

    shim = types.ModuleType("torchmcubes")
    shim.marching_cubes = marching_cubes
    sys.modules["torchmcubes"] = shim


_VIT_RENAMES = [
    (r"\.encoder\.layer\.(\d+)\.attention\.attention\.query\.", r".layers.\1.attention.q_proj."),
    (r"\.encoder\.layer\.(\d+)\.attention\.attention\.key\.", r".layers.\1.attention.k_proj."),
    (r"\.encoder\.layer\.(\d+)\.attention\.attention\.value\.", r".layers.\1.attention.v_proj."),
    (r"\.encoder\.layer\.(\d+)\.attention\.output\.dense\.", r".layers.\1.attention.o_proj."),
    (r"\.encoder\.layer\.(\d+)\.intermediate\.dense\.", r".layers.\1.mlp.fc1."),
    (r"\.encoder\.layer\.(\d+)\.output\.dense\.", r".layers.\1.mlp.fc2."),
    (r"\.encoder\.layer\.(\d+)\.layernorm_(before|after)\.", r".layers.\1.layernorm_\2."),
]


def remap_vit_keys(state: dict, model_keys: set) -> dict:
    """Translate old transformers ViT weight names to the names the installed version expects.

    The TripoSR checkpoint uses `encoder.layer.N.attention.attention.query`; newer
    transformers releases call the same weights `layers.N.attention.q_proj`. A key is
    only renamed when the model does not already know it, so this is a no-op on
    versions that still use the old names.
    """
    import re

    out = {}
    for key, value in state.items():
        if key not in model_keys:
            for pattern, repl in _VIT_RENAMES:
                new_key = re.sub(pattern, repl, key)
                if new_key != key and new_key in model_keys:
                    key = new_key
                    break
        out[key] = value
    return out


class TripoSRReconstructor:
    @staticmethod
    def _load_model(TSR, r: dict):
        import torch
        from huggingface_hub import hf_hub_download
        from omegaconf import OmegaConf

        config_path = hf_hub_download(repo_id=r["triposr_repo"], filename="config.yaml")
        weight_path = hf_hub_download(repo_id=r["triposr_repo"], filename="model.ckpt")
        conf = OmegaConf.load(config_path)
        OmegaConf.resolve(conf)
        model = TSR(conf)
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(remap_vit_keys(state, set(model.state_dict().keys())))
        return model

    def __init__(self, cfg: dict) -> None:
        import torch

        _ensure_marching_cubes()
        # TripoSR code is cloned into third_party/TripoSR by setup.sh
        repo = ROOT / "third_party" / "TripoSR"
        if not repo.exists():
            raise RuntimeError("third_party/TripoSR is missing. Run `bash setup.sh` first.")
        sys.path.insert(0, str(repo))
        from tsr.system import TSR  # noqa: E402

        r = cfg["reconstruct"]
        self.cfg = r
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self._load_model(TSR, r)
        self.model.renderer.set_chunk_size(int(r["chunk_size"]))
        self.model.to(self.device)

    def reconstruct(self, rgba: Image.Image, mc_resolution: int | None = None):
        """Return a trimesh.Trimesh with vertex colours. Higher mc_resolution = finer mesh."""
        # TripoSR expects the object composited on a mid-grey background.
        arr = np.array(rgba.convert("RGBA")).astype(np.float32) / 255.0
        rgb = arr[..., :3] * arr[..., 3:4] + (1 - arr[..., 3:4]) * 0.5
        image = Image.fromarray((rgb * 255).astype(np.uint8))

        with self.torch.no_grad():
            codes = self.model([image], device=self.device)
            meshes = self.model.extract_mesh(
                codes, has_vertex_color=True, resolution=int(mc_resolution or self.cfg["mc_resolution"])
            )
        return meshes[0]


def get_reconstructor(cfg: dict):
    name = cfg["reconstruct"]["model"].lower()
    if name == "triposr":
        return TripoSRReconstructor(cfg)
    raise ValueError(f"Unknown reconstruction model '{name}'. Available: triposr")
