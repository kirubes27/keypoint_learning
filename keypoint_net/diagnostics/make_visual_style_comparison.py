"""Preserve old diagnostic visuals, regenerate corrected visuals, compare."""

from pathlib import Path

from PIL import Image, ImageDraw

import day1_channel_suite
from dxutils import OUTPUTS


MODELS = ("task80", "smoke64", "smoke128")


def _preserve_before() -> None:
    for model in MODELS:
        for stem in (f"day1_overlay_{model}_frame45", f"day1_heatmaps_{model}_frame45"):
            source = OUTPUTS / f"{stem}.png"
            target = OUTPUTS / f"style_before_{stem}.png"
            assert source.exists(), source
            Image.open(source).save(target)


def _side_by_side(before: Path, after: Path, output: Path, title: str) -> None:
    left = Image.open(before).convert("RGB")
    right = Image.open(after).convert("RGB")
    height = max(left.height, right.height)
    if left.height != height:
        left = left.resize((round(left.width * height / left.height), height))
    if right.height != height:
        right = right.resize((round(right.width * height / right.height), height))
    header = 42
    canvas = Image.new("RGB", (left.width + right.width, height + header), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 3), title, fill="black")
    draw.text((8, 21), "BEFORE: hollow rings / independent heatmap scaling", fill="black")
    draw.text((left.width + 8, 21), "AFTER: original dots / shared absolute probability scale", fill="black")
    canvas.paste(left, (0, header))
    canvas.paste(right, (left.width, header))
    canvas.save(output)


def main() -> None:
    _preserve_before()
    day1_channel_suite.main()
    for model in MODELS:
        _side_by_side(
            OUTPUTS / f"style_before_day1_overlay_{model}_frame45.png",
            OUTPUTS / f"day1_overlay_{model}_frame45.png",
            OUTPUTS / f"style_comparison_overlay_{model}_frame45.png",
            f"{model}: standard RGB keypoint overlay",
        )
        _side_by_side(
            OUTPUTS / f"style_before_day1_heatmaps_{model}_frame45.png",
            OUTPUTS / f"day1_heatmaps_{model}_frame45.png",
            OUTPUTS / f"style_comparison_heatmaps_{model}_frame45.png",
            f"{model}: heatmap probability visualization",
        )
    print(f"Wrote before/after comparisons to {OUTPUTS}")


if __name__ == "__main__":
    main()
