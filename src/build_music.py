"""Generate calm ambient soundtrack for the Reel.

20-second piece in A minor, ~70 BPM. Layered soft sine pads, plucked
piano-like melody, subtle low warmth and a quiet wind/air bed —
matching the "framed window view" aesthetic.
"""
from __future__ import annotations

import math
import struct
import wave
from pathlib import Path

import numpy as np

SR = 44100
DURATION = 22.0
N = int(SR * DURATION)
t = np.linspace(0.0, DURATION, N, endpoint=False)


def hz(midi: float) -> float:
    return 440.0 * (2.0 ** ((midi - 69) / 12.0))


def adsr(length: int, attack=0.01, decay=0.2, sustain=0.6, release=0.4) -> np.ndarray:
    a = int(SR * attack)
    d = int(SR * decay)
    r = int(SR * release)
    s = max(0, length - a - d - r)
    env = np.concatenate([
        np.linspace(0, 1, a, endpoint=False),
        np.linspace(1, sustain, d, endpoint=False),
        np.full(s, sustain),
        np.linspace(sustain, 0, r, endpoint=False),
    ])
    if len(env) < length:
        env = np.pad(env, (0, length - len(env)))
    return env[:length]


def pluck(freq: float, start: float, dur: float, gain: float = 0.25) -> np.ndarray:
    """Soft piano-like pluck: fundamental + 2 harmonics, fast decay."""
    out = np.zeros(N)
    s = int(start * SR)
    L = int(dur * SR)
    if s + L > N:
        L = N - s
    if L <= 0:
        return out
    local_t = np.arange(L) / SR
    env = np.exp(-3.0 * local_t) * (1 - np.exp(-80 * local_t))
    wave_ = (
        1.00 * np.sin(2 * np.pi * freq * local_t)
        + 0.45 * np.sin(2 * np.pi * 2 * freq * local_t)
        + 0.18 * np.sin(2 * np.pi * 3 * freq * local_t)
        + 0.08 * np.sin(2 * np.pi * 4 * freq * local_t)
    )
    out[s : s + L] = wave_ * env * gain
    return out


def pad(freqs: list[float], start: float, dur: float, gain: float = 0.12) -> np.ndarray:
    """Sustained sine-stack pad with slow chorus-like detune."""
    out = np.zeros(N)
    s = int(start * SR)
    L = int(dur * SR)
    if s + L > N:
        L = N - s
    if L <= 0:
        return out
    local_t = np.arange(L) / SR
    env = adsr(L, attack=0.8, decay=0.5, sustain=0.85, release=1.5)
    sig = np.zeros(L)
    for f in freqs:
        sig += np.sin(2 * np.pi * f * local_t)
        sig += 0.6 * np.sin(2 * np.pi * (f * 1.003) * local_t + 0.3)
        sig += 0.3 * np.sin(2 * np.pi * (f * 0.5) * local_t)
    sig /= max(1.0, len(freqs))
    out[s : s + L] = sig * env * gain
    return out


def air_bed(gain: float = 0.012) -> np.ndarray:
    """Very low filtered noise — like air/breath in a quiet room."""
    rng = np.random.default_rng(7)
    noise = rng.standard_normal(N)
    # simple 1-pole low-pass (cutoff ~ 200 Hz)
    alpha = 1 - math.exp(-2 * math.pi * 200 / SR)
    y = np.zeros_like(noise)
    acc = 0.0
    for i in range(N):
        acc += alpha * (noise[i] - acc)
        y[i] = acc
    # slow tremolo
    trem = 0.7 + 0.3 * np.sin(2 * np.pi * 0.07 * t)
    return y * trem * gain


def stereo(mono: np.ndarray, pan: float = 0.0) -> np.ndarray:
    """pan in [-1, 1]."""
    p = (pan + 1) / 2
    left = mono * math.cos(p * math.pi / 2)
    right = mono * math.sin(p * math.pi / 2)
    return np.stack([left, right], axis=1)


# --- Composition --------------------------------------------------------
# A minor: A=57, C=60, E=64, G=67
# Chord progression (each ~5.5s): Am — Fmaj7 — Cmaj9 — Em7
chords = [
    ("Am",   [hz(57), hz(60), hz(64), hz(69)]),
    ("Fmaj7",[hz(53), hz(57), hz(60), hz(64)]),
    ("Cmaj9",[hz(48), hz(55), hz(59), hz(62)]),
    ("Em7",  [hz(52), hz(55), hz(59), hz(62)]),
]

mix = np.zeros((N, 2))

# Pads (chord bed)
for i, (_, freqs) in enumerate(chords):
    start = i * 5.5
    dur = 6.0  # overlap into next chord
    pad_mono = pad(freqs, start, dur, gain=0.18)
    mix += stereo(pad_mono, pan=-0.15) * 0.6
    mix += stereo(pad_mono, pan=+0.15) * 0.6

# Melody plucks (sparse, contemplative — pentatonic A minor)
# notes (midi, time, dur)
melody = [
    (72, 0.5, 1.6),  # C5
    (76, 1.8, 1.4),  # E5
    (74, 3.0, 1.0),  # D5
    (72, 4.0, 1.2),  # C5

    (69, 5.8, 2.0),  # A4
    (74, 7.2, 1.6),  # D5
    (72, 8.6, 1.4),  # C5

    (76, 10.5, 1.4),  # E5  (gentle climax)
    (79, 11.8, 1.6),  # G5
    (76, 13.2, 1.6),  # E5

    (74, 15.0, 1.4),  # D5
    (72, 16.2, 1.4),  # C5
    (69, 17.4, 2.4),  # A4
    (67, 19.4, 2.4),  # G4 — outro
]
for midi, st, du in melody:
    m = pluck(hz(midi), st, du, gain=0.22)
    mix += stereo(m, pan=+0.05)

# Low warm bass — root note per chord
for i, (_, freqs) in enumerate(chords):
    root_low = freqs[0] / 2
    b = pluck(root_low, i * 5.5 + 0.05, 5.5, gain=0.18)
    mix += stereo(b, pan=0.0) * 0.9

# Air bed (subtle texture)
ab = air_bed(gain=0.02)
mix += stereo(ab, pan=-0.4) * 0.5
mix += stereo(ab, pan=+0.4) * 0.5

# Soft global fade-in / fade-out
fade_in = int(SR * 0.8)
fade_out = int(SR * 1.6)
env = np.ones(N)
env[:fade_in] = np.linspace(0, 1, fade_in)
env[-fade_out:] = np.linspace(1, 0, fade_out)
mix *= env[:, None]

# Soft tanh saturation + normalization
mix = np.tanh(mix * 1.1)
peak = np.max(np.abs(mix))
if peak > 0:
    mix = mix / peak * 0.92

# Write 16-bit PCM WAV
out_path = Path(__file__).resolve().parent.parent / "output" / "music.wav"
out_path.parent.mkdir(parents=True, exist_ok=True)
samples = (mix * 32767.0).astype(np.int16)
with wave.open(str(out_path), "wb") as w:
    w.setnchannels(2)
    w.setsampwidth(2)
    w.setframerate(SR)
    w.writeframes(samples.tobytes())

print(f"wrote {out_path}  ({DURATION:.1f}s, {SR} Hz)")
