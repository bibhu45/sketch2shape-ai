import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from sketch2shape.generate import build_prompt  # noqa: E402
from sketch2shape.metrics import StageTimer, fidelity_score  # noqa: E402
from sketch2shape.config import load_config  # noqa: E402


def test_fidelity_high_when_edges_match_and_low_when_not():
    sketch = np.zeros((256, 256), np.uint8)
    cv2.rectangle(sketch, (60, 60), (200, 200), 255, 3)
    matching = np.full((256, 256, 3), 255, np.uint8)
    cv2.rectangle(matching, (60, 60), (200, 200), (0, 0, 0), 3)
    unrelated = np.full((256, 256, 3), 255, np.uint8)
    cv2.circle(unrelated, (30, 30), 12, (0, 0, 0), 3)
    s = Image.fromarray(sketch)
    assert fidelity_score(s, Image.fromarray(matching)) > 0.8
    assert fidelity_score(s, Image.fromarray(unrelated)) < 0.1


def test_empty_prompt_still_builds_a_prompt():
    cfg = load_config()
    pos, neg = build_prompt("", cfg)
    assert pos and neg
    assert "red mug" in build_prompt("red mug", cfg)[0]


def test_stage_timer():
    t = StageTimer()
    with t.stage("a"):
        pass
    assert "a" in t.times and t.total >= 0
