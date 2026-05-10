"""Compose the 1080x1920 vertical Instagram Reel.

Concept: 「絵画を飾ったような窓の外の景色」
The window is the frame, the forest is the painting. We stage each photo
as a piece hung on a softly-lit gallery wall — generous breathing room,
a faint drop-shadow, no text — so each shot can be looked at, not read.

Music: assets/Tea_Leaves_and_Ink.mp3 — full length used, scene cuts are
timed against it.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

W, H = 1080, 1920
FPS = 30

# Soft warm "gallery wall" base color (slightly darker than paper-white;
# a vertical gradient overlay applied later gives depth).
WALL_TOP = (242, 235, 220)
WALL_BOT = (218, 209, 191)


# --- Scene definitions --------------------------------------------------
# Each photo is held on screen ~5s — long enough to read the composition
# the way you would a hung painting. No text, anywhere.
SCENES = [
    {"img": "7C1A4171.JPG", "dur": 5.0, "kb": (1.00, 1.04, 0.0,  -0.005)},
    {"img": "7C1A4172.JPG", "dur": 5.0, "kb": (1.04, 1.00, 0.0,  +0.005)},
    {"img": "7C1A4173.JPG", "dur": 5.0, "kb": (1.04, 1.00, -0.005, 0.0)},
    {"img": "7C1A4184.JPG", "dur": 5.0, "kb": (1.00, 1.04, 0.0,  -0.005)},
    {"img": "7C1A4182.JPG", "dur": 5.0, "kb": (1.04, 1.00, +0.005, 0.0)},
    {"img": "7C1A4174.JPG", "dur": 5.5, "kb": (1.00, 1.05, -0.005, -0.005)},
]
TOTAL = sum(s["dur"] for s in SCENES)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def make_wall() -> Image.Image:
    """Soft vertical gradient wall, like indirect light from above.

    A faint paper-grain texture is added so the surface does not look
    flat-digital."""
    top = np.array(WALL_TOP, dtype=np.float32)
    bot = np.array(WALL_BOT, dtype=np.float32)
    grad = np.linspace(0.0, 1.0, H, dtype=np.float32) ** 1.2
    col = top + (bot - top) * grad[:, None]
    arr = np.broadcast_to(col[:, None, :], (H, W, 3)).copy()

    # very subtle warm-white grain
    rng = np.random.default_rng(11)
    grain = rng.standard_normal((H, W, 1)).astype(np.float32) * 1.6
    arr = np.clip(arr + grain, 0, 255)
    return Image.fromarray(arr.astype(np.uint8))


def fit_contain(img: Image.Image, w: int, h: int) -> tuple[Image.Image, int, int]:
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    return img.resize((nw, nh), Image.LANCZOS), (w - nw) // 2, (h - nh) // 2


def lift_shadows(arr: np.ndarray) -> np.ndarray:
    """Lift shadows so the dark interiors read clearly without blowing
    out the bright window. Plus a hint of warmth for gallery light."""
    arr = np.power(arr, 0.78)
    arr = (arr - 0.5) * 1.04 + 0.52
    warm = np.array([[[+0.012, +0.006, -0.006]]], dtype=np.float32)
    arr = arr + warm
    return np.clip(arr, 0.0, 1.0)


def kenburns_crop(base: Image.Image, kb, t01: float) -> Image.Image:
    """Slow Ken Burns inside the source — preserves native aspect."""
    z0, z1, dx, dy = kb
    e = smoothstep(t01)
    z = z0 + (z1 - z0) * e
    px = dx * e
    py = dy * e
    bw, bh = base.size
    cw = bw / z
    ch = bh / z
    cx = bw / 2 + px * bw
    cy = bh / 2 + py * bh
    left = max(0, int(cx - cw / 2))
    top = max(0, int(cy - ch / 2))
    right = min(bw, int(cx + cw / 2))
    bottom = min(bh, int(cy + ch / 2))
    return base.crop((left, top, right, bottom))


def hang_photo(wall_with_shadow: Image.Image, photo_target: tuple[int, int, int, int],
               photo: Image.Image) -> Image.Image:
    """Paste the (already toned + sized) photo onto a wall that already
    carries the drop-shadow for this scene."""
    canvas = wall_with_shadow.copy()
    x, y, _, _ = photo_target
    canvas.paste(photo, (x, y))
    return canvas


def precompute_scene(wall: Image.Image, photo_aspect: float) -> tuple[Image.Image, tuple[int, int, int, int]]:
    """For a given source aspect ratio, build the wall+shadow once and
    return the fixed (x, y, w, h) where each per-frame photo will land."""
    margin_x = int(W * 0.08)
    margin_y = int(H * 0.10)
    inner_w = W - 2 * margin_x
    inner_h = H - 2 * margin_y

    # Fit-contain bounds for this aspect
    if photo_aspect >= inner_w / inner_h:
        fw = inner_w
        fh = max(1, int(round(inner_w / photo_aspect)))
    else:
        fh = inner_h
        fw = max(1, int(round(inner_h * photo_aspect)))
    x = (W - fw) // 2
    y = (H - fh) // 2

    base = wall.convert("RGBA")
    shadow = Image.new("RGBA", (fw + 80, fh + 80), (0, 0, 0, 0))
    sd = ImageDraw.Draw(shadow)
    sd.rectangle([40, 40, 40 + fw, 40 + fh], fill=(0, 0, 0, 110))
    shadow = shadow.filter(ImageFilter.GaussianBlur(24))
    base.alpha_composite(shadow, (x - 40 + 8, y - 40 + 14))
    return base.convert("RGB"), (x, y, fw, fh)


def build_frames() -> Path:
    tmp = OUTPUT / "_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    wall = make_wall()

    full_bases = []
    scene_walls = []
    scene_targets = []
    for s in SCENES:
        im = Image.open(ASSETS / s["img"]).convert("RGB")
        # 1800-px long-side keeps quality but trims memory considerably
        scale = 1800 / max(im.size)
        if scale < 1:
            im = im.resize(
                (int(im.size[0] * scale), int(im.size[1] * scale)),
                Image.LANCZOS,
            )
        full_bases.append(im)
        sw, st = precompute_scene(wall, im.size[0] / im.size[1])
        scene_walls.append(sw)
        scene_targets.append(st)

    def render(si: int, t01: float) -> Image.Image:
        crop = kenburns_crop(full_bases[si], SCENES[si]["kb"], t01)
        x, y, fw, fh = scene_targets[si]
        crop = crop.resize((fw, fh), Image.LANCZOS)
        arr = np.asarray(crop).astype(np.float32) / 255.0
        arr = lift_shadows(arr)
        crop = Image.fromarray((arr * 255).astype(np.uint8))
        return hang_photo(scene_walls[si], scene_targets[si], crop)

    frame_idx = 0
    xfade = 0.8
    for si, scene in enumerate(SCENES):
        n = int(round(scene["dur"] * FPS))
        for k in range(n):
            t01 = k / max(1, n - 1)
            scene_t = k / FPS

            f = render(si, t01)

            if si < len(SCENES) - 1 and scene_t > scene["dur"] - xfade:
                tail = scene_t - (scene["dur"] - xfade)
                mix = smoothstep(tail / xfade)
                next_f = render(si + 1, mix * 0.04)
                f = Image.blend(f, next_f, mix)
                del next_f

            f.save(tmp / f"f_{frame_idx:05d}.jpg", quality=90)
            del f
            frame_idx += 1

            if k % 30 == 0:
                print(f"scene {si+1}/{len(SCENES)}  frame {frame_idx}", flush=True)
        print(f"scene {si+1}/{len(SCENES)} done ({n} frames)", flush=True)

    print(f"total frames: {frame_idx}")
    return tmp


def encode_video(frames_dir: Path) -> Path:
    out = OUTPUT / "kanoya_may_course_reel.mp4"
    music = ASSETS / "Tea_Leaves_and_Ink.mp3"
    fade_out = 1.8
    afilter = (
        f"afade=t=in:st=0:d=0.8,"
        f"afade=t=out:st={TOTAL - fade_out:.2f}:d={fade_out}"
    )
    cmd = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", str(frames_dir / "f_%05d.jpg"),
        "-i", str(music),
        "-af", afilter,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-preset", "medium",
        "-crf", "20",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        "-movflags", "+faststart",
        str(out),
    ]
    subprocess.run(cmd, check=True)
    return out


if __name__ == "__main__":
    frames = build_frames()
    mp4 = encode_video(frames)
    print(f"wrote {mp4}")
