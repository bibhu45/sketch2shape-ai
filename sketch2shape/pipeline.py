"""Chains all stages. The CLI and the Gradio app both call `Pipeline.run`."""
from __future__ import annotations

import json
import time
from pathlib import Path

from PIL import Image

from .config import load_config
from .metrics import StageTimer, fidelity_details, gpu_info
from .preprocess import SketchError, detect_input_kind, load_sketch, prepare_picture, preprocess


class PipelineError(RuntimeError):
    """A failure with a message that is safe to show to the user."""


class Pipeline:
    def __init__(self, cfg: dict | None = None, low_vram: bool = False) -> None:
        self.cfg = cfg or load_config()
        self.low_vram = low_vram
        self._gen = None
        self._rec = None

    # Models load lazily and stay resident, so only the first run pays the cost.
    def _generator(self):
        if self._gen is None:
            from .generate import Generator

            self._gen = Generator(self.cfg, low_vram=self.low_vram)
        return self._gen

    def _reconstructor(self):
        if self._rec is None:
            from .reconstruct import get_reconstructor

            self._rec = get_reconstructor(self.cfg)
        return self._rec

    def _preset(self, quality: str) -> dict:
        presets = self.cfg.get("quality_presets", {})
        return presets.get(quality) or presets.get("Balanced") or {
            "steps": self.cfg["generation"]["steps"],
            "candidates": 1,
            "mc_resolution": self.cfg["reconstruct"]["mc_resolution"],
        }

    def run(
        self,
        sketch,
        prompt: str | None = None,
        out_dir: str | Path | None = None,
        seed: int | None = None,
        sketch_strength: float | None = None,
        steps: int | None = None,
        mode: str = "auto",
        quality: str = "Balanced",
        orientation: str | None = None,
        smooth: bool = True,
        progress=None,
        on_image=None,
    ) -> dict:
        """Run the whole pipeline.

        mode: "auto" (detect), "sketch" (line drawing, redrawn by the diffusion model),
              "picture" (photo or colour image, sent straight to 3D for best accuracy).
        progress(message) and on_image(key, PIL.Image) report live status. Keys:
        "input", "reference", "cutout".
        """
        say = progress or (lambda msg: None)
        show = on_image or (lambda key, image: None)
        cfg, timer = self.cfg, StageTimer()
        preset = self._preset(quality)
        orientation = orientation or cfg["mesh"].get("orientation", "Auto (recommended)")
        out = Path(out_dir or Path(cfg["paths"]["outputs"]) / time.strftime("run-%Y%m%d-%H%M%S"))
        out.mkdir(parents=True, exist_ok=True)
        base_seed = int(cfg["seed"] if seed is None else seed)
        fidelity = None
        retried = False
        candidate_scores: list[float] = []

        try:
            say("Checking your input")
            with timer.stage("preprocess"):
                original = load_sketch(sketch)
                detected = detect_input_kind(original)
                used_mode = detected if mode == "auto" else mode
                if used_mode not in ("sketch", "picture"):
                    raise SketchError(f"Unknown input type '{mode}'.")
                if used_mode == "sketch":
                    scribble, pre_info = preprocess(original, size=cfg["image_size"])
                    show("input", scribble)
                else:
                    reference = prepare_picture(original)
                    pre_info = {"input_size": list(original.size)}
                    show("input", reference)
            original.save(out / "input_sketch.png")

            if used_mode == "sketch":
                scribble.save(out / "sketch_processed.png")
                n = max(1, int(preset["candidates"]))
                gen = self._generator()
                best = None
                with timer.stage("generate"):
                    for i in range(n):
                        say(f"Drawing the object from your sketch (try {i + 1} of {n})")
                        image = gen.generate(scribble, prompt, base_seed + i, sketch_strength, steps or preset["steps"])
                        score = fidelity_details(scribble, image)
                        candidate_scores.append(score["score"])
                        if best is None or score["score"] > best[0]:
                            best = (score["score"], image)
                            show("reference", image)
                    if best[0] < cfg["generation"]["min_fidelity"]:
                        retried = True
                        say("Result drifted from your sketch, retrying with stronger sketch guidance")
                        image = gen.generate(
                            scribble, prompt, base_seed, cfg["generation"]["retry_strength"], steps or preset["steps"]
                        )
                        score = fidelity_details(scribble, image)
                        candidate_scores.append(score["score"])
                        if score["score"] > best[0]:
                            best = (score["score"], image)
                            show("reference", image)
                fidelity, reference = best
            else:
                scribble = None
                show("reference", reference)
            reference.save(out / "generated.png")

            say("Removing the background")
            from .prepare import prepare

            with timer.stage("prepare"):
                cutout = prepare(
                    reference,
                    cfg["prepare"]["foreground_ratio"],
                    cfg["image_size"],
                    cfg["prepare"].get("min_part_share", 0.05),
                    cfg["prepare"].get("erode_px", 1),
                )
            cutout.save(out / "cutout.png")
            show("cutout", cutout)

            say("Building the 3D model")
            with timer.stage("reconstruct"):
                mesh = self._reconstructor().reconstruct(cutout, mc_resolution=preset["mc_resolution"])

            say("Cleaning, checking and exporting the mesh")
            from .mesh import clean_mesh, export_and_verify, orient_mesh, smooth_mesh, validate_mesh

            with timer.stage("mesh"):
                mesh = orient_mesh(mesh, orientation)
                mesh = clean_mesh(mesh, cfg["mesh"]["keep_largest_component_share"])
                if smooth:
                    mesh = smooth_mesh(mesh, cfg["mesh"].get("smooth_iterations", 10))
                validation = validate_mesh(mesh, cfg)
                export = export_and_verify(mesh, out, cfg)
        except SketchError as err:
            raise PipelineError(str(err)) from err
        except Exception as err:
            msg = str(err)
            if "out of memory" in msg.lower():
                msg = (
                    "The GPU ran out of memory. Restart the runtime, choose the Fast quality, "
                    "or lower reconstruct.chunk_size in config.yaml."
                )
            raise PipelineError(msg) from err

        ok = bool(validation["passed"] and export["reload_ok"] and export["size_ok"])
        result = {
            "success": ok,
            "glb": str(out / "model.glb"),
            "obj": str(out / "model.obj"),
            "generated_image": str(out / "generated.png"),
            "prompt": prompt or "",
            "seed": base_seed,
            "mode_requested": mode,
            "mode_used": used_mode,
            "auto_detected": detected,
            "quality": quality,
            "orientation": orientation,
            "fidelity": fidelity,
            "candidate_scores": candidate_scores,
            "fidelity_retry_used": retried,
            "validation": validation,
            "export": export,
            "timings_sec": timer.times,
            "total_sec": timer.total,
            "preprocess": pre_info,
            "hardware": gpu_info(),
        }
        (out / "metadata.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
        (out / "validation_report.json").write_text(
            json.dumps({"validation": validation, "export": export}, indent=2), encoding="utf-8"
        )
        return result
