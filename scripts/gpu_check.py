"""Run this first in a Kaggle or Colab notebook: python gpu_check.py

Confirms the GPU, VRAM, internet access and the key libraries, and prints
the numbers you need for the README hardware section.
"""
import importlib
import platform
import shutil
import subprocess
import sys
import urllib.request


def line(label, value):
    print(f"{label:<22}{value}")


print("=== Environment ===")
line("Python", sys.version.split()[0])
line("OS", platform.platform())

print("\n=== GPU ===")
try:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    line("nvidia-smi", out)
except Exception:
    line("nvidia-smi", "NOT FOUND -> turn on the GPU accelerator in notebook settings")

try:
    import torch

    line("PyTorch", torch.__version__)
    line("CUDA available", torch.cuda.is_available())
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        line("GPU name", props.name)
        line("VRAM (GB)", round(props.total_memory / 1024**3, 1))
        line("CUDA version", torch.version.cuda)
    else:
        print("-> No CUDA GPU visible. The heavy models will not run.")
except ImportError:
    line("PyTorch", "not installed (pip install torch)")

print("\n=== Disk and internet ===")
total, used, free = shutil.disk_usage(".")
line("Free disk (GB)", round(free / 1024**3, 1))
for name, url in [("Hugging Face", "https://huggingface.co"), ("GitHub", "https://github.com"), ("PyPI", "https://pypi.org")]:
    try:
        urllib.request.urlopen(url, timeout=8)
        line(name, "reachable")
    except Exception as exc:
        line(name, f"NOT reachable ({type(exc).__name__}) -> enable Internet in notebook settings")

print("\n=== Libraries ===")
for mod in ["diffusers", "transformers", "cv2", "PIL", "trimesh", "rembg", "gradio"]:
    try:
        m = importlib.import_module(mod)
        line(mod, getattr(m, "__version__", "installed"))
    except ImportError:
        line(mod, "missing")
