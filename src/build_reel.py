"""Compose the 1080x1920 vertical Instagram Reel for 奈良春日 鹿のや＜KANOYA＞.

Aesthetic: "絵画を飾ったような窓の外の景色" — let the photos breathe.
Each photo keeps its native aspect on a softly-blurred bed of itself,
shadows are lifted hard so the rooms read clearly, and only two
understated wordmarks (奈良春日　鹿のや / KANOYA) bookend the spot.
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
# and an optional caption (only opener + closer carry text).
SCENES = [
    {
        "img": "7C1A4171.JPG",
        "dur": 4.0,
        "kb": (1.00, 1.06, 0.0, -0.01),
        "brand_jp": "奈良春日　鹿のや",
        "brand_en": "KANOYA",
        "tagline": None,
    },
    {
        "img": "7C1A4172.JPG",
        "dur": 3.6,
        "kb": (1.05, 1.00, 0.0, 0.01),
    },
    {
        "img": "7C1A4173.JPG",
        "dur": 3.6,
        "kb": (1.06, 1.00, -0.01, 0.0),
    },
    {
        "img": "7C1A4184.JPG",
        "dur": 3.6,
        "kb": (1.00, 1.06, 0.0, -0.01),
    },
    {
        "img": "7C1A4182.JPG",
        "dur": 3.8,
        "kb": (1.06, 1.00, 0.01, 0.0),
    },
    {
        "img": "7C1A4174.JPG",
        "dur": 3.6,
        "kb": (1.00, 1.05, -0.01, -0.01),
        "brand_jp": "奈良春日　鹿のや",
        "brand_en": "KANOYA",
        "tagline": "窓のむこうに、初夏。",
    },
]
TOTAL = sum(s["dur"] for s in SCENES)


def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, x))
    return x * x * (3 - 2 * x)


def fit_contain(img: Image.Image, w: int, h: int) -> tuple[Image.Image, int, int]:
    """Resize so the whole image fits inside w x h. Returns (image, x, y)."""
    iw, ih = img.size
    scale = min(w / iw, h / ih)
    nw, nh = max(1, int(round(iw * scale))), max(1, int(round(ih * scale)))
    resized = img.resize((nw, nh), Image.LANCZOS)
    x = (w - nw) // 2
    y = (h - nh) // 2
    return resized, x, y


def fit_cover(img: Image.Image, w: int, h: int) -> Image.Image:
    """Resize to fully cover w x h while preserving aspect (used for the bg)."""
    iw, ih = img.size
    scale = max(w / iw, h / ih)
    nw, nh = int(math.ceil(iw * scale)), int(math.ceil(ih * scale))
    return img.resize((nw, nh), Image.LANCZOS)


def lift_shadows(arr: np.ndarray) -> np.ndarray:
    """Brighten the image by lifting shadows (gamma) plus mild contrast.

    Source photos are intentionally moody (dark room, bright window) so
    a multiplicative boost would clip the highlights. A gamma < 1 lifts
    the midtones / shadows while leaving the brightest pixels alone.

    This pass goes further than the previous version — the dark interior
    walls/floors now sit comfortably in mid-tones, while the bright
    window foliage still carries detail (clipped only at 1.0).
    """
    arr = np.power(arr, 0.55)            # aggressive shadow lift
    arr = (arr - 0.5) * 0.88 + 0.58      # gentler contrast, brighter midpoint
    arr = arr + 0.06                     # global lift
    # gallery-light warmth
    warm = np.array([[[+0.014, +0.007, -0.007]]], dtype=np.float32)
    arr = arr + warm
    return np.clip(arr, 0.0, 1.0)


def render_photo(base_full: Image.Image, base_bg: Image.Image, kb, t01: float) -> Image.Image:
    """Render one frame: blurred bg + contained foreground with a gentle Ken Burns.

    base_full is the source photo at native aspect (already moderately resized).
    base_bg is a pre-built blurred 1080x1920 background derived from the same photo.
    """
    z0, z1, dx, dy = kb
    e = smoothstep(t01)
    z = z0 + (z1 - z0) * e
    px = dx * e
    py = dy * e

    # Foreground: take the full-aspect photo, scale-zoom around its center,
    # then fit_contain into the frame.
    bw, bh = base_full.size
    cw = bw / z
    ch = bh / z
    cx = bw / 2 + px * bw
    cy = bh / 2 + py * bh
    left = max(0, int(cx - cw / 2))
    top = max(0, int(cy - ch / 2))
    right = min(bw, int(cx + cw / 2))
    bottom = min(bh, int(cy + ch / 2))
    crop = base_full.crop((left, top, right, bottom))

    fg, fx, fy = fit_contain(crop, W, H)

    canvas = base_bg.copy()
    canvas.paste(fg, (fx, fy))
    return canvas


def build_blur_bg(base_full: Image.Image) -> Image.Image:
    """Make a soft, brightened, blurred 1080x1920 backdrop from the same photo.

    The fitted foreground overlays this — so any visible bands at top/bottom
    feel like a soft extension of the image rather than a hard letterbox.
    """
    bg = fit_cover(base_full, W, H)
    iw, ih = bg.size
    bx = (iw - W) // 2
    by = (ih - H) // 2
    bg = bg.crop((bx, by, bx + W, by + H))
    bg = bg.filter(ImageFilter.GaussianBlur(60))
    arr = np.asarray(bg).astype(np.float32) / 255.0
    # very bright and low-contrast so the bands feel like soft daylight
    arr = np.power(arr, 0.42)
    arr = (arr - 0.5) * 0.38 + 0.68
    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8))


def text_with_shadow(canvas, xy, text, font, fill=(255, 255, 255), alpha=1.0, anchor="mm"):
    if alpha <= 0 or not text:
        return
    layer = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    ld = ImageDraw.Draw(layer)
    for off, a in [(6, 80), (3, 120), (1, 160)]:
        ld.text(
            (xy[0] + off // 2, xy[1] + off // 2),
            text, font=font, fill=(0, 0, 0, a), anchor=anchor,
        )
    layer = layer.filter(ImageFilter.GaussianBlur(2.5))
    fd = ImageDraw.Draw(layer)
    r, g, b = fill
    fd.text(xy, text, font=font, fill=(r, g, b, int(255 * alpha)), anchor=anchor)
    canvas.alpha_composite(layer)


def caption_layer(
    scene_t: float, scene_dur: float,
    brand_jp: str | None = None,
    brand_en: str | None = None,
    tagline: str | None = None,
) -> Image.Image:
    """Three-tier KANOYA wordmark: tagline / 鹿のや / KANOYA. Fades softly."""
    layer = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    if not (brand_jp or brand_en or tagline):
        return layer

    fade_in_t = 0.6
    fade_out_t = 0.8
    if scene_t < fade_in_t:
        a = scene_t / fade_in_t
    elif scene_t > scene_dur - fade_out_t:
        a = max(0.0, (scene_dur - scene_t) / fade_out_t)
    else:
        a = 1.0
    a = smoothstep(a)

    # Layout (anchored toward the bottom):
    #   tagline      → small, above
    #   brand_jp     → main, "奈良春日　鹿のや"
    #   brand_en     → spaced KANOYA underneath
    cy_en = int(H * 0.92)
    cy_jp = cy_en - 78
    cy_tag = cy_jp - 86

    if brand_en:
        en_font = ImageFont.truetype(FONT_BOLD, 36)
        spaced_en = "  ".join(list(brand_en))
        text_with_shadow(
            layer, (W // 2, cy_en), spaced_en, en_font,
            fill=(252, 248, 235), alpha=a * 0.95, anchor="mm",
        )
    if brand_jp:
        jp_font = ImageFont.truetype(FONT_BOLD, 56)
        text_with_shadow(
            layer, (W // 2, cy_jp), brand_jp, jp_font,
            fill=(252, 248, 235), alpha=a, anchor="mm",
        )
    if tagline:
        tag_font = ImageFont.truetype(FONT_BOLD, 36)
        text_with_shadow(
            layer, (W // 2, cy_tag), tagline, tag_font,
            fill=(238, 232, 216), alpha=a * 0.88, anchor="mm",
        )
    return layer


def build_frames() -> Path:
    tmp = OUTPUT / "_frames"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir()

    full_bases = []
    blur_bgs = []
    for s in SCENES:
        im = Image.open(ASSETS / s["img"]).convert("RGB")
        # downscale long side to ~2400 for speed; aspect preserved
        scale = 2400 / max(im.size)
        if scale < 1:
            im = im.resize(
                (int(im.size[0] * scale), int(im.size[1] * scale)),
                Image.LANCZOS,
            )
        full_bases.append(im)
        blur_bgs.append(build_blur_bg(im))

    frame_idx = 0
    for si, scene in enumerate(SCENES):
        n = int(round(scene["dur"] * FPS))
        for k in range(n):
            t01 = k / max(1, n - 1)
            scene_t = k / FPS

            f = render_photo(full_bases[si], blur_bgs[si], scene["kb"], t01)

            # Cross-fade between scenes (0.6s)
            xfade = 0.6
            if si < len(SCENES) - 1 and scene_t > scene["dur"] - xfade:
                tail = scene_t - (scene["dur"] - xfade)
                mix = tail / xfade
                next_f = render_photo(
                    full_bases[si + 1], blur_bgs[si + 1],
                    SCENES[si + 1]["kb"], 0.0 + mix * 0.04,
                )
                f = Image.blend(f, next_f, smoothstep(mix))

            # Brighten + tone (lift shadows, mild warmth, no vignette)
            arr = np.asarray(f).astype(np.float32) / 255.0
            arr = lift_shadows(arr)
            arr = (arr * 255.0).astype(np.uint8)
            f = Image.fromarray(arr).convert("RGBA")

            cap = caption_layer(
                scene_t, scene["dur"],
                brand_jp=scene.get("brand_jp"),
                brand_en=scene.get("brand_en"),
                tagline=scene.get("tagline"),
            )
            f.alpha_composite(cap)

            f = f.convert("RGB")
            f.save(tmp / f"f_{frame_idx:05d}.jpg", quality=92)
            frame_idx += 1
        print(f"scene {si+1}/{len(SCENES)} rendered ({n} frames)")

    print(f"total frames: {frame_idx}")
    return tmp


def encode_video(frames_dir: Path) -> Path:
    out = OUTPUT / "kanoya_may_course_reel.mp4"
    music = ASSETS / "Tea_Leaves_and_Ink.mp3"
    fade_out = 1.5
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
