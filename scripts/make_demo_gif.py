"""Render docs/assets/demo.gif from a real `annotrust scan` run.

Pillow is not a runtime dependency. Install it only to regenerate the GIF:

    python -m pip install pillow
    python scripts/make_demo_gif.py
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/jetbrains-mono/JetBrainsMono-Regular.ttf",
    "/usr/share/fonts/truetype/macos/JetBrainsMono-Regular.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
)
BG = (22, 26, 32)
FG = (220, 224, 230)
GREEN = (126, 186, 137)
RED = (232, 140, 140)
DIM = (150, 160, 175)
YELLOW = (214, 178, 102)


def main() -> None:
    lines = _demo_lines()
    font = ImageFont.truetype(_font_path(), 13)
    line_height = 18
    pad = 16
    visible = 24
    char_width = font.getlength("M")
    longest = max(len(line) for line in lines)
    width = pad * 2 + int(char_width * (longest + 1))
    height = 44 + visible * line_height + 12
    frames: list[Image.Image] = []
    durations: list[int] = []
    # Reveal the command block, then scroll the report.
    counts = list(range(4, len(lines) + 1, 2))
    if not counts or counts[-1] != len(lines):
        counts.append(len(lines))
    for count in counts:
        offset = max(0, count - visible)
        frames.append(_frame(lines[offset:count], font, width, height, line_height, pad, offset > 0))
        durations.append(280)
    frames.append(frames[-1].copy())
    durations.append(1600)
    out = ROOT / "docs" / "assets" / "demo.gif"
    out.parent.mkdir(parents=True, exist_ok=True)
    palette = [frame.convert("P", palette=Image.Palette.ADAPTIVE, colors=32) for frame in frames]
    palette[0].save(
        out,
        save_all=True,
        append_images=palette[1:],
        duration=durations,
        loop=0,
        optimize=True,
    )
    print(f"wrote {out} ({out.stat().st_size} bytes, {len(frames)} frames)")


def _demo_lines() -> list[str]:
    work = ROOT / ".demo-tmp"
    if work.exists():
        for child in work.iterdir():
            child.unlink()
    else:
        work.mkdir()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    simulate = [
        sys.executable,
        "-m",
        "annotrust",
        "simulate",
        "--out",
        "demo.csv",
        "--truth",
        "demo.truth.json",
        "--seed",
        "7",
        "--class-names",
        "cat,dog,bird,fish",
        "--poison-source",
        "cat",
        "--poison-target",
        "dog",
    ]
    scan = [
        sys.executable,
        "-m",
        "annotrust",
        "scan",
        "demo.csv",
        "--json",
        "demo.json",
        "--html",
        "demo.html",
        "--fail-on",
        "high",
    ]
    sim = subprocess.run(simulate, cwd=work, env=env, check=True, capture_output=True, text=True)
    scanned = subprocess.run(scan, cwd=work, env=env, check=False, capture_output=True, text=True)
    if scanned.returncode != 1:
        raise SystemExit(f"expected exit 1, got {scanned.returncode}\n{scanned.stderr}")
    lines = [
        "$ python -m annotrust simulate --out demo.csv --truth demo.truth.json \\",
        "    --seed 7 --class-names cat,dog,bird,fish --poison-source cat --poison-target dog",
        *sim.stdout.splitlines(),
        "",
        "$ python -m annotrust scan demo.csv --json demo.json --html demo.html --fail-on high",
        *scanned.stdout.splitlines(),
        "$ echo $?",
        "1",
    ]
    return lines


def _frame(
    visible_lines: list[str],
    font: ImageFont.FreeTypeFont,
    width: int,
    height: int,
    line_height: int,
    pad: int,
    scrolled: bool,
) -> Image.Image:
    image = Image.new("RGB", (width, height), BG)
    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, width, 28), fill=(32, 36, 44))
    draw.text((pad, 6), "annotrust  demo.csv", font=font, fill=DIM)
    top = 36
    for index, line in enumerate(visible_lines):
        color = FG
        if line.startswith("$"):
            color = GREEN
        elif "FAIL" in line or line == "1":
            color = RED
        elif line.startswith("planted:") or line.startswith("annotator severity:"):
            color = YELLOW
        draw.text((pad, top + index * line_height), line, font=font, fill=color)
    if scrolled:
        draw.text((width - pad - 80, 6), "scrolled", font=font, fill=DIM)
    return image


def _font_path() -> str:
    for candidate in FONT_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
    raise SystemExit("no monospace font found")


if __name__ == "__main__":
    main()
