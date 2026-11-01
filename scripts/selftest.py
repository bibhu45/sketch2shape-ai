"""End-to-end self-test: python scripts/selftest.py
Runs the full pipeline on the bundled sample sketch and prints PASS or FAIL.
Judges can run this once to confirm the setup works.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sketch2shape.pipeline import Pipeline, PipelineError  # noqa: E402

SAMPLE = Path(__file__).resolve().parent.parent / "samples" / "inputs" / "mug.png"


def main() -> int:
    if not SAMPLE.exists():
        print(f"FAIL: sample sketch missing at {SAMPLE}")
        return 1
    try:
        r = Pipeline().run(SAMPLE, prompt="a red ceramic mug", progress=print)
    except PipelineError as err:
        print(f"FAIL: {err}")
        return 1
    print(f"Time: {r['total_sec']} s  Fidelity: {r['fidelity']}  Faces: {r['validation']['faces']}")
    print("PASS" if r["success"] else f"FAIL: validation {r['validation']['checks']}")
    return 0 if r["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
