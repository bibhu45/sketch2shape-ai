"""Command line: python cli.py --input samples/inputs/mug.png --prompt "red ceramic mug" """
import argparse
import json
import sys

from sketch2shape.config import load_config
from sketch2shape.pipeline import Pipeline, PipelineError


def main() -> int:
    p = argparse.ArgumentParser(description="Sketch to textured 3D model.")
    p.add_argument("--input", required=True, help="Path to the sketch image")
    p.add_argument("--prompt", default="", help="Optional text prompt")
    p.add_argument("--out", default=None, help="Output folder (default: outputs/run-<timestamp>)")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--sketch-strength", type=float, default=None, help="ControlNet strength")
    p.add_argument("--steps", type=int, default=None)
    p.add_argument("--low-vram", action="store_true")
    p.add_argument("--config", default=None)
    args = p.parse_args()

    pipe = Pipeline(load_config(args.config), low_vram=args.low_vram)
    try:
        result = pipe.run(
            args.input,
            prompt=args.prompt,
            out_dir=args.out,
            seed=args.seed,
            sketch_strength=args.sketch_strength,
            steps=args.steps,
            progress=print,
        )
    except PipelineError as err:
        print(f"\nFailed: {err}", file=sys.stderr)
        return 1

    print(json.dumps({k: result[k] for k in ("success", "glb", "fidelity", "timings_sec", "total_sec", "hardware")}, indent=2))
    return 0 if result["success"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
