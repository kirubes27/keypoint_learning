"""Recreate the sweep's five-frame visualization style from cached outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from dxutils import OUTPUTS, draw_keypoints, frame_files, to_px


MODELS = ("task80", "smoke64", "smoke128")
FRAME_INDICES = (0, 44, 88, 132, 176)
REFERENCE_TASK80 = Path(
    "/Users/kirubeso.r/Documents/PhD/cluster_downloads/"
    "hammer_full360_shared_complete/keypoint_net/runs_hammer_full360_shared/"
    "phase_a_engineers_hammer_vray_20260606_151123_941396_seed42_pid3289780/"
    "visualizations/sequence.png"
)


def _font(size: int):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size=size)
    except OSError:
        return ImageFont.load_default()


def sweep_sequence(coords: np.ndarray, output: Path) -> None:
    """Five square panels and titles, matching visualize_sequence()."""
    panel = 400
    title_height = 34
    gap = 10
    canvas = Image.new(
        "RGB",
        (panel * len(FRAME_INDICES) + gap * (len(FRAME_INDICES) - 1), panel + title_height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    title_font = _font(20)
    files = frame_files()
    pixels = to_px(coords)
    for position, frame_index in enumerate(FRAME_INDICES):
        x_offset = position * (panel + gap)
        frame = Image.open(files[frame_index]).convert("RGB").resize((panel, panel))
        frame = draw_keypoints(frame, pixels[frame_index])
        text = f"Frame {frame_index}"
        text_box = draw.textbbox((0, 0), text, font=title_font)
        text_width = text_box[2] - text_box[0]
        draw.text(
            (x_offset + (panel - text_width) / 2, 4),
            text,
            fill="black",
            font=title_font,
        )
        canvas.paste(frame, (x_offset, title_height))
    canvas.save(output)


def vertical_comparison(reference: Path, recreated: Path, output: Path) -> None:
    top = Image.open(reference).convert("RGB")
    bottom = Image.open(recreated).convert("RGB")
    width = max(top.width, bottom.width)
    if top.width != width:
        top = top.resize((width, round(top.height * width / top.width)))
    if bottom.width != width:
        bottom = bottom.resize((width, round(bottom.height * width / bottom.width)))
    header = 38
    canvas = Image.new("RGB", (width, top.height + bottom.height + 2 * header), "white")
    draw = ImageDraw.Draw(canvas)
    font = _font(20)
    draw.text((10, 7), "REFERENCE: original sweep visualization", fill="black", font=font)
    canvas.paste(top, (0, header))
    second_y = header + top.height
    draw.text((10, second_y + 7), "RECREATED: diagnostic output in sweep style", fill="black", font=font)
    canvas.paste(bottom, (0, second_y + header))
    canvas.save(output)


def main() -> None:
    for model in MODELS:
        cache = np.load(OUTPUTS / f"day1_cache_{model}.npz")
        output = OUTPUTS / f"day1_sequence_sweep_style_{model}.png"
        sweep_sequence(cache["coords"], output)
        print(f"saved {output.name}")
    assert REFERENCE_TASK80.exists(), REFERENCE_TASK80
    vertical_comparison(
        REFERENCE_TASK80,
        OUTPUTS / "day1_sequence_sweep_style_task80.png",
        OUTPUTS / "style_comparison_actual_sweep_vs_diagnostic_task80.png",
    )
    print("saved style_comparison_actual_sweep_vs_diagnostic_task80.png")


if __name__ == "__main__":
    main()
