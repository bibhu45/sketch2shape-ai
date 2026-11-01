#!/usr/bin/env bash
# One-command setup for Google Colab or Kaggle (Linux + NVIDIA GPU).
# Usage: bash setup.sh
# Note: pip "dependency resolver" warnings about numba, pytensor, cudf or cuml are harmless on Colab.
set -e

echo "== Installing Python dependencies =="
pip install -q -r requirements.txt

echo "== Fetching TripoSR code =="
if [ ! -d third_party/TripoSR ]; then
  git clone https://github.com/VAST-AI-Research/TripoSR third_party/TripoSR
fi

# We do NOT install TripoSR's own requirements.txt:
#  - it pins old versions (e.g. transformers) that clash with diffusers on Colab
#  - it needs torchmcubes, which compiles from source and fails on newer CUDA setups.
# Instead we install only what the reconstruction code needs. Marching cubes comes from
# scikit-image through a shim in sketch2shape/reconstruct.py.
echo "== Installing TripoSR runtime packages =="
pip install -q omegaconf einops scikit-image imageio
pip install -q xatlas moderngl || echo "(optional: xatlas/moderngl skipped, only needed for texture baking)"

echo "== Done. Next: python scripts/gpu_check.py, then python scripts/selftest.py =="
