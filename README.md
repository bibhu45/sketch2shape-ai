# Sketch2Shape AI

Turn a single 2D sketch (drawn in the app or uploaded), with an optional text prompt, into a
textured 3D model exported as `.glb` and `.obj`.

> TechZephyr 4.0 hackathon project. Fill every `TODO` before submitting.

## 1. Objective
TODO: 2 to 3 sentences. Input: one sketch + optional prompt. Output: a textured 3D mesh that keeps the sketch's structure.

## 2. Architecture
```
Sketch/upload -> Preprocess -> Sketch-guided image (SD + ControlNet) -> Background removal
   -> Image-to-3D (TripoSR) -> Mesh clean-up + validation -> Export .glb/.obj -> Preview + download
```
TODO: add docs/architecture.png (the diagram from the SRS).

| Stage | Module | Technique |
| --- | --- | --- |
| Preprocess | `sketch2shape/preprocess.py` | OpenCV: denoise, CLAHE, adaptive threshold, auto-invert, pad, resize |
| Generate | `sketch2shape/generate.py` | Stable Diffusion 1.5 + ControlNet Scribble, UniPC scheduler |
| Prepare | `sketch2shape/prepare.py` | rembg, crop, centre |
| Reconstruct | `sketch2shape/reconstruct.py` | TripoSR (baseline). TODO: InstantMesh if added |
| Mesh | `sketch2shape/mesh.py` | trimesh: clean, normalise, validate, export, reload check |

## 3. Models, APIs and third-party components (disclosure)
TODO: verify each licence on its page and fill in. No external API is used at run time.

| Component | Role | Source | Licence |
| --- | --- | --- | --- |
| Stable Diffusion 1.5 | image generation | Hugging Face `stable-diffusion-v1-5/stable-diffusion-v1-5` | TODO |
| ControlNet 1.1 Scribble | sketch conditioning | Hugging Face `lllyasviel/control_v11p_sd15_scribble` | TODO |
| rembg (isnet-general-use) | background removal | PyPI | TODO |
| TripoSR | image-to-3D | GitHub `VAST-AI-Research/TripoSR` | TODO |
| diffusers, transformers, trimesh, gradio, OpenCV, Pillow | libraries | PyPI | TODO |

## 4. Hardware and runtime
TODO: fill from `python scripts/gpu_check.py` and a real run.

| Item | Value |
| --- | --- |
| GPU used | TODO (e.g. Colab T4, 16 GB) |
| Cold start (model loading) | TODO s |
| Warm run, sketch to .glb | TODO s |
| Peak GPU memory | TODO GB |

A CUDA GPU is required. There is no CPU path.

## 5. Setup and run (Google Colab, recommended)
1. Open `notebooks/colab_run.ipynb` in Colab. Runtime > Change runtime type > T4 GPU.
2. Set `REPO` in the first cell to this repository's URL, then run the cells in order.
3. Cell 5 runs the self-test and should print `PASS`.
4. Cell 6a starts the web app and prints a public link.

Manual setup on any Linux machine with an NVIDIA GPU:
```bash
git clone <this repo> && cd sketch2shape-ai
bash setup.sh
python scripts/gpu_check.py
python scripts/selftest.py
python app.py            # web app
python cli.py --input samples/inputs/mug.png --prompt "a red ceramic mug"
```
Model weights download automatically from Hugging Face on first run (several GB, needs internet once).

## 6. Usage
- **Web app:** draw or upload a sketch, add an optional prompt, click Generate. Rotate the 3D preview, download `.glb` or `.obj`.
- **Prompt is optional.** With no prompt the system uses a neutral default and still produces a model.
- **Input type:** *Auto-detect* tells a line drawing from a colour picture. Sketches are redrawn by Stable Diffusion + ControlNet (best of N tries, scored against your lines); pictures skip the redraw and go straight to 3D for maximum fidelity.
- **Quality:** Fast (1 try, 192 mesh), Balanced (3 tries, 256), Best (4 tries, 320).
- **Orientation:** if a model appears sideways, change *Advanced > Model orientation*.
- **CLI:** `python cli.py --input <sketch> [--prompt "..."] [--out folder] [--seed N]`.

## 7. Inputs and outputs
Input: PNG, JPG, JPEG, WEBP or BMP, up to 10 MB, at least 128 px per side.
Each run writes a folder (`outputs/run-<time>/`): `input_sketch.png`, `sketch_processed.png`, `generated.png`,
`cutout.png`, `model.glb`, `model.obj`, `validation_report.json`, `metadata.json` (timings, GPU, fidelity, mesh stats).

## 8. Validation
The system reloads the exported files and checks: opens, non-empty geometry, finite values, real depth
(not a flat plane), colour or texture present, file size. See `sketch2shape/mesh.py`.

## 9. Results
TODO: add 3 to 5 sketch / generated image / 3D result examples from `samples/outputs/` and Blender screenshots.

## 10. Limitations
- A single sketch cannot define hidden sides. Back and underside geometry and texture are inferred.
- TripoSR gives vertex-coloured meshes, so textures are simpler than a UV-mapped texture.
- Thin parts, transparent objects and multi-object sketches reconstruct poorly.
- Very abstract sketches can be reinterpreted by the diffusion stage. Use the sketch-strength slider.
- Meshes are not guaranteed watertight and are not print-ready or animation-ready.
- Needs an NVIDIA GPU and internet for the first run. Results can vary slightly between machines.
TODO: add anything you observe during testing.

## 11. Troubleshooting
- **No CUDA GPU:** Colab > Runtime > Change runtime type > T4 GPU, then restart.
- **Out of memory:** restart the runtime, lower `reconstruct.mc_resolution` or `chunk_size` in `config.yaml`, or pass `--low-vram`.
- **torchmcubes build fails:** not needed. setup.sh skips it and `sketch2shape/reconstruct.py` uses a scikit-image marching-cubes replacement.
- **Model download denied:** accept the licence on the model's Hugging Face page and set `HF_TOKEN`.

## 12. Tests
`python -m pytest -q tests`
