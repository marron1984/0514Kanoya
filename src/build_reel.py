"""Compose the 1080x1920 vertical Instagram Reel.

Aesthetic: "絵画を飾ったような窓の外の景色" — the room frames a living
painting. We render frames with PIL (slow Ken-Burns + cinematic letterbox-
free vertical fit + soft text fade-in) and pipe to ffmpeg, then mux the
soundtrack from output/music.wav.
"""
from __future__ import annotations

import math
import shutil
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent.parent
ASSETS = ROOT / "assets"
OUTPUT = ROOT / "output"
OUTPUT.mkdir(exist_ok=True)

W, H = 1080, 1920
FPS = 30

FONT_BOLD = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc"
FONT_REG = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"


# --- Scene definitions --------------------------------------------------
# Each scene: image, duration (s), kenburns (start_zoom, end_zoom, dx, dy),
# and the captions (main, sub) shown during it.
SCENES = [
    {
        "img": "7C1A4171.JPG",
        "dur": 4.0,
        "kb": (1.05, 1.18, 0.0, -0.02),
        "main": "窓を、額に。",
        "sub": "5月｜月替わりコース",
    },
    {
        "img": "7C1A4172.JPG",
        "dur": 3.6,
        "kb": (1.10, 1.00, 0.0, 0.03),
        "main": "新緑が、",
        "sub": "今日の献立を告げる。",
    },
    {
        "img": "7C1A4173.JPG",
        "dur": 3.6,
        "kb": (1.18, 1.05, -0.02, 0.0),
        "main": "旬の鮮魚を、",
        "sub": "確かな腕で。",
    },
    {
        "img": "7C1A4184.JPG",
        "dur": 3.6,
        "kb": (1.05, 1.18, 0.0, -0.02),
        "main": "ひと皿に、",
        "sub": "鹿屋の風景を盛る。",
    },
    {
        "img": "7C1A4182.JPG",
        "dur": 3.8,
        "kb": (1.18, 1.06, 0.03, 0.0),
        "main": "目で愉しみ、",
        "sub": "舌で旅する。",
    },
    {
        "img": "7C1A4174.JPG",
        "dur": 3.4,
        "kb": (1.10, 1.20, -0.02, -0.02),
        "main": "May Course",
        "sub": "保存して、その日まで。",
    },
]
TOTAL = sum(s["dur"] for s in SCENES)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fit_cover(img: Image.Image, w: int, h: int) -> Image.Image:
    """Resize so it fully covers w x h while preserving aspect."""
    iw, ih = img.size
    scale = max(w / iw, h / ih)
    nw, nh = int(math.ceil(iw * scale)), int(math.ceil(ih * scale))
    return img.resize((nw, nh), Image.LANCZOS)


def kenburns_frame(base: Image.Image, kb, t01: float) -> Image.Image:
    """Render one frame with a slow zoom + tiny pan within base image."""
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
    crop = base.crop((left, top, right, bottom))
    return crop.resize((W, H), Image.LANCZOS)


def soft_vignette(img: Image.Image, strength: float = 0.55) -> Image.Image:
    """Subtle dark vignette toward edges to push focus inward."""
    w, h = img.size
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)
    # radial gradient by ellipses
    steps = 60
    for i in range(steps):
        r = i / steps
        alpha = int(255 * (1 - r) ** 1.6)
        bw = int(w * (0.55 + 0.55 * r))
        bh = int(h * (0.55 + 0.55 * r))
        x0 = (w - bw) // 2
        y0 = (h - bh) // 2
        d.ellipse([x0, y0, x0 + bw, y0 + bh], fill=alpha)
    mask = mask.filter(ImageFilter.GaussianBlur(120))
    dark = Image.new("RGB", (w, h), (0, 0, 0))
    out = Image.composite(img, dark, mask)
    return Image.blend(img, out, strength)


def text_with_shadow(
    canvas: Image.Image,
    xy,
    text: str,
    font: ImageFont.FreeTypeFont,
    fill=(255, 255, 255),
    alpha: float = 1.0,
    anchor: str = "lm",
) -> None:
    if alpha <= 0 or not text:
        return
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    # soft drop shadow (multi-pass)
    for off, a in [(8, 90), (4, 130), (2, 170)]:
        ld.text(
            (xy[0] + off // 2, xy[1] + off // 2),
            text,
            font=font,
            fill=(0, 0, 0, a),
            anchor=anchor,
        )
    layer = layer.filter(ImageFilter.GaussianBlur(3))
    fd = ImageDraw.Draw(layer)
    r, g, b = fill
    fd.text(xy, text, font=font, fill=(r, g, b, int(255 * alpha)), anchor=anchor)
    canvas.alpha_composite(layer)


def caption_layer(scene_t: float, scene_dur: float, main: str, sub: str) -> Image.Image:
    """Return RGBA caption overlay for the given moment within a scene."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))

    # caption fade: in 0.4s, hold, out 0.6s
    fade_in_t = 0.4
    fade_out_t = 0.6
    if scene_t < fade_in_t:
        a = scene_t / fade_in_t
    elif scene_t > scene_dur - fade_out_t:
        a = max(0.0, (scene_dur - scene_t) / fade_out_t)
    else:
        a = 1.0
    a = smoothstep(a)

    # subtle gradient at the bottom for legibility
    grad = Image.new("L", (1, H), 0)
    for y in range(H):
        if y > H * 0.55:
            v = int(220 * smoothstep((y - H * 0.55) / (H * 0.45)))
            grad.putpixel((0, y), v)
    grad = grad.resize((W, H))
    grad_rgba = Image.merge("RGBA", (
        Image.new("L", (W, H), 0),
        Image.new("L", (W, H), 0),
        Image.new("L", (W, H), 0),
        grad,
    ))
    layer.alpha_composite(grad_rgba)

    main_font = ImageFont.truetype(FONT_BOLD, 86)
    sub_font = ImageFont.truetype(FONT_REG, 48)

    cy_main = int(H * 0.78)
    cy_sub = cy_main + 110

    text_with_shadow(
        layer, (W // 2, cy_main), main, main_font,
        fill=(252, 248, 235), alpha=a, anchor="mm",
    )
    text_with_shadow(
        layer, (W // 2, cy_sub), sub, sub_font,
        fill=(232, 228, 215), alpha=a * 0.95, anchor="mm",
    )
    return layer


def brand_layer(t: float) -> Image.Image:
    """Top-left small brand mark, fades in at 0.6s and stays."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    a = smoothstep(min(1.0, max(0.0, (t - 0.6) / 0.8)))
    if a <= 0:
        return layer
    f = ImageFont.truetype(FONT_REG, 34)
    text_with_shadow(
        layer, (60, 70), "KANOYA  ｜  May Course",
        f, fill=(245, 240, 225), alpha=a * 0.85, anchor="lm",
    )
    return layer


def build_frames() -> Path:
    """Render all frames into a temp folder and return its path."""
    tmp = OUTPUT / "_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    # Pre-load + pre-fit each scene base
    bases = []
    for s in SCENES:
        im = Image.open(ASSETS / s["img"]).convert("RGB")
        # downscale long side to ~2400 for speed but still quality
        scale = 2400 / max(im.size)
        if scale < 1:
            im = im.resize(
                (int(im.size[0] * scale), int(im.size[1] * scale)),
                Image.LANCZOS,
            )
        bases.append(fit_cover(im, W, H))

    frame_idx = 0
    global_t = 0.0
    for si, scene in enumerate(SCENES):
        n = int(round(scene["dur"] * FPS))
        for k in range(n):
            t01 = k / max(1, n - 1)
            scene_t = k / FPS

            f = kenburns_frame(bases[si], scene["kb"], t01)
            f = soft_vignette(f, strength=0.5)

            # Cross-fade between scenes (0.5s)
            xfade = 0.5
            if si < len(SCENES) - 1 and scene_t > scene["dur"] - xfade:
                next_base = bases[si + 1]
                next_kb = SCENES[si + 1]["kb"]
                tail = scene_t - (scene["dur"] - xfade)
                mix = tail / xfade
                next_f = kenburns_frame(next_base, next_kb, 0.0 + mix * 0.05)
                next_f = soft_vignette(next_f, strength=0.5)
                f = Image.blend(f, next_f, smoothstep(mix))

            # Cinematic film-grade tint: slight green-cyan in shadows,
            # warm in highlights — like a museum gallery.
            arr = np.asarray(f).astype(np.float32) / 255.0
            lum = arr.mean(axis=2, keepdims=True)
            shadows = np.clip(1 - lum, 0, 1)
            highlights = np.clip(lum, 0, 1)
            tint_shadow = np.array([[[ -0.02, +0.015, +0.02 ]]])
            tint_high   = np.array([[[ +0.03, +0.015, -0.015 ]]])
            arr = arr + shadows * tint_shadow + highlights * tint_high
            # gentle contrast
            arr = (arr - 0.5) * 1.05 + 0.5
            arr = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
            f = Image.fromarray(arr).convert("RGBA")

            cap = caption_layer(scene_t, scene["dur"], scene["main"], scene["sub"])
            f.alpha_composite(cap)

            br = brand_layer(global_t)
            f.alpha_composite(br)

            # final flatten
            f = f.convert("RGB")
            f.save(tmp / f"f_{frame_idx:05d}.jpg", quality=92)

            frame_idx += 1
            global_t += 1.0 / FPS
        print(f"scene {si+1}/{len(SCENES)} rendered ({n} frames)")

    print(f"total frames: {frame_idx}")
    return tmp


def encode_video(frames_dir: Path) -> Path:
    out = OUTPUT / "kanoya_may_course_reel.mp4"
    # BGM: repository-provided MP3, faded to match video length.
    music = ASSETS / "Tea_Leaves_and_Ink.mp3"
    fade_out = 1.5  # seconds
    afilter = f"afade=t=in:st=0:d=0.6,afade=t=out:st={TOTAL - fade_out:.2f}:d={fade_out}"
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
