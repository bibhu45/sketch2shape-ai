"""Stage 3: sketch-guided image generation (Stable Diffusion + scribble ControlNet)."""
from __future__ import annotations

from PIL import Image


def build_prompt(user_prompt: str | None, cfg: dict) -> tuple[str, str]:
    """The prompt is optional. An empty prompt falls back to a neutral default."""
    g = cfg["generation"]
    base = (user_prompt or "").strip() or g["default_prompt"]
    return base[:300] + g["prompt_suffix"], g["negative_prompt"]


class Generator:
    """Loads the diffusion models once; call .generate() as often as needed."""

    def __init__(self, cfg: dict, low_vram: bool = False) -> None:
        import torch
        from diffusers import (
            ControlNetModel,
            StableDiffusionControlNetPipeline,
            UniPCMultistepScheduler,
        )

        if not torch.cuda.is_available():
            raise RuntimeError("No CUDA GPU found. In Colab: Runtime > Change runtime type > T4 GPU.")

        self.cfg = cfg
        self.torch = torch
        g = cfg["generation"]
        controlnet = ControlNetModel.from_pretrained(g["controlnet"], torch_dtype=torch.float16)
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            g["base_model"],
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        if low_vram:
            pipe.enable_model_cpu_offload()
        else:
            pipe.to("cuda")
        pipe.enable_attention_slicing()
        self.pipe = pipe

    def generate(
        self,
        scribble: Image.Image,
        prompt: str | None,
        seed: int | None = None,
        strength: float | None = None,
        steps: int | None = None,
    ) -> Image.Image:
        g = self.cfg["generation"]
        pos, neg = build_prompt(prompt, self.cfg)
        generator = self.torch.Generator(device="cpu").manual_seed(
            int(self.cfg["seed"] if seed is None else seed)
        )
        result = self.pipe(
            pos,
            image=scribble,
            negative_prompt=neg,
            num_inference_steps=int(steps or g["steps"]),
            guidance_scale=g["guidance_scale"],
            controlnet_conditioning_scale=float(strength if strength is not None else g["sketch_strength"]),
            generator=generator,
        )
        return result.images[0]

    def free(self) -> None:
        """Release GPU memory before the reconstruction stage if VRAM is tight."""
        del self.pipe
        self.torch.cuda.empty_cache()
