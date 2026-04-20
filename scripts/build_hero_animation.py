#!/usr/bin/env python3
"""Generate docs/assets/hero_kuramoto.svg — README hero animation.

Minimalist, elegant Kuramoto phase-synchronization animation.
Embeds SMIL-animated SVG compatible with GitHub markdown rendering.

Theme: N phase oscillators drift, then lock. Order parameter R(t) crosses gate.
"""

from __future__ import annotations

import math
from itertools import pairwise
from pathlib import Path

# Canvas
W, H = 1280, 400

# Palette — aligned with neurophase badge colors.
BG = "#0B0F1A"           # deep near-black
GRID = "#16203A"          # subtle
MUTED = "#3A4A66"         # desync dot
INDIGO = "#1F5082"        # brand primary (mypy badge)
VIOLET = "#8B3FBF"        # brand accent (blueviolet status)
EMERALD = "#00C853"       # doctor green, READY state
GOLD = "#E8C547"          # R vector tip
TEXT = "#E8EAED"
TEXT_DIM = "#9AA3B2"

# Left disc: oscillator ring
DISC_CX, DISC_CY, DISC_R = 220, 200, 120
N = 14                    # oscillators

# Right panel: R(t) trace
TRACE_X0, TRACE_X1 = 500, 1200
TRACE_Y0, TRACE_Y1 = 60, 340   # y0=top (R=1), y1=bottom (R=0)

# Animation timing
DUR = 9.0                 # seconds per full cycle
KEY_TIMES = [0.0, 0.22, 0.50, 0.78, 1.0]  # desync → lock → drift → lock → desync

# --- phase schedule per oscillator ----------------------------------------
# At keyframe 0 (desync): phases uniformly spread + jitter → R ≈ 0.12
# At keyframe 1 (locking): phases clustered around common mean → R ≈ 0.88
# At keyframe 2 (partial drift): slight rotation, moderate cluster → R ≈ 0.55
# At keyframe 3 (re-lock): tight cluster, rotated mean → R ≈ 0.92
# At keyframe 4 = keyframe 0 (loop)

def jitter(i: int, seed: int) -> float:
    # deterministic pseudo-jitter in [-1, 1]
    x = math.sin((i + 1) * 17.13 + seed * 2.71) * 1e4
    return (x - math.floor(x)) * 2 - 1

def phase_schedule(i: int) -> list[float]:
    """Return 5 phase values (radians) at KEY_TIMES for oscillator i."""
    spread_base = 2 * math.pi * i / N
    natural = 0.04 * jitter(i, 1)                 # tiny natural-frequency jitter
    desync = spread_base + 0.15 * jitter(i, 2)
    lock1_mean = math.pi * 0.5
    lock1 = lock1_mean + 0.18 * jitter(i, 3)
    drift = lock1_mean + 0.9 * math.pi * 0.3 + 0.55 * jitter(i, 4)
    lock2_mean = math.pi * 1.15
    lock2 = lock2_mean + 0.14 * jitter(i, 5)
    # wrap to desync at end of loop (same as frame 0 but advance by 2π to avoid
    # back-rotation jump): add 2π so SMIL rotates forward
    end = desync + 2 * math.pi + natural * 0
    return [desync, lock1, drift, lock2, end]


def dot_positions(phase: float) -> tuple[float, float]:
    return (
        DISC_CX + DISC_R * math.cos(phase),
        DISC_CY + DISC_R * math.sin(phase),
    )


def order_parameter(phases: list[float]) -> tuple[float, float]:
    """R, psi from complex mean of e^{iθ}."""
    sx = sum(math.cos(p) for p in phases) / len(phases)
    sy = sum(math.sin(p) for p in phases) / len(phases)
    return math.hypot(sx, sy), math.atan2(sy, sx)


# --- build schedules -------------------------------------------------------
schedules = [phase_schedule(i) for i in range(N)]

R_keyframes: list[tuple[float, float]] = []  # (R, psi) per keyframe
for k in range(len(KEY_TIMES)):
    phases_k = [schedules[i][k] for i in range(N)]
    R_keyframes.append(order_parameter(phases_k))

# Flatten gate threshold
THRESHOLD = 0.65
GATE_Y = TRACE_Y1 - (TRACE_Y1 - TRACE_Y0) * THRESHOLD


# --- SVG helpers -----------------------------------------------------------
def fmt(x: float) -> str:
    return f"{x:.3f}".rstrip("0").rstrip(".")


def smil_values(values: list[float]) -> str:
    return ";".join(fmt(v) for v in values)


KEY_TIMES_STR = ";".join(fmt(t) for t in KEY_TIMES)


# --- build SVG -------------------------------------------------------------
parts: list[str] = []
parts.append(
    f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" '
    f'width="100%" role="img" aria-label="neurophase — Kuramoto phase synchronization animation" '
    f'style="max-width:100%;height:auto;font-family:ui-monospace,SFMono-Regular,Menlo,monospace">'
)

# defs: gradients
parts.append(
    """
<defs>
  <radialGradient id="disc-glow" cx="50%" cy="50%" r="50%">
    <stop offset="0%" stop-color="#1F5082" stop-opacity="0.22"/>
    <stop offset="55%" stop-color="#1F5082" stop-opacity="0.06"/>
    <stop offset="100%" stop-color="#0B0F1A" stop-opacity="0"/>
  </radialGradient>
  <linearGradient id="trace-grad" x1="0" x2="1" y1="0" y2="0">
    <stop offset="0%" stop-color="#8B3FBF" stop-opacity="0.0"/>
    <stop offset="10%" stop-color="#8B3FBF" stop-opacity="0.9"/>
    <stop offset="100%" stop-color="#00C853" stop-opacity="0.95"/>
  </linearGradient>
  <filter id="soft-glow" x="-50%" y="-50%" width="200%" height="200%">
    <feGaussianBlur stdDeviation="2.4" result="b"/>
    <feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge>
  </filter>
</defs>
"""
)

# background
parts.append(f'<rect width="{W}" height="{H}" fill="{BG}"/>')

# subtle top rule
parts.append(
    f'<line x1="60" y1="34" x2="{W-60}" y2="34" stroke="{GRID}" stroke-width="1"/>'
)

# title block
parts.append(
    f'<text x="60" y="24" fill="{TEXT}" font-size="13" font-weight="600" '
    f'letter-spacing="3">NEUROPHASE</text>'
)
parts.append(
    f'<text x="{W-60}" y="24" fill="{TEXT_DIM}" font-size="11" text-anchor="end" '
    f'letter-spacing="2">KURAMOTO · R(t) · GATE</text>'
)

# --- LEFT: oscillator disc ------------------------------------------------
parts.append(f'<circle cx="{DISC_CX}" cy="{DISC_CY}" r="{DISC_R+28}" fill="url(#disc-glow)"/>')
# ring
parts.append(
    f'<circle cx="{DISC_CX}" cy="{DISC_CY}" r="{DISC_R}" fill="none" '
    f'stroke="{GRID}" stroke-width="1"/>'
)
# axes (very subtle)
parts.append(
    f'<line x1="{DISC_CX-DISC_R-8}" y1="{DISC_CY}" x2="{DISC_CX+DISC_R+8}" y2="{DISC_CY}" '
    f'stroke="{GRID}" stroke-width="0.6" stroke-dasharray="2 4"/>'
)
parts.append(
    f'<line x1="{DISC_CX}" y1="{DISC_CY-DISC_R-8}" x2="{DISC_CX}" y2="{DISC_CY+DISC_R+8}" '
    f'stroke="{GRID}" stroke-width="0.6" stroke-dasharray="2 4"/>'
)

# oscillators: each a dot whose (cx, cy) animates
for i in range(N):
    cx_vals = [DISC_CX + DISC_R * math.cos(p) for p in schedules[i]]
    cy_vals = [DISC_CY + DISC_R * math.sin(p) for p in schedules[i]]
    # colour animation: interpolate MUTED → INDIGO → MUTED → INDIGO → MUTED
    # by animating opacity on a top "lock" dot overlaid on a base "muted" dot.
    # Simpler: animate fill between two colours via multiple animate tags.
    parts.append(
        f'<circle r="5.2" fill="{INDIGO}" opacity="0.95" filter="url(#soft-glow)">'
        f'<animate attributeName="cx" values="{smil_values(cx_vals)}" '
        f'keyTimes="{KEY_TIMES_STR}" dur="{DUR}s" repeatCount="indefinite" '
        f'calcMode="spline" keySplines="0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1"/>'
        f'<animate attributeName="cy" values="{smil_values(cy_vals)}" '
        f'keyTimes="{KEY_TIMES_STR}" dur="{DUR}s" repeatCount="indefinite" '
        f'calcMode="spline" keySplines="0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1"/>'
        f'<animate attributeName="fill" values="{MUTED};{INDIGO};{MUTED};{INDIGO};{MUTED}" '
        f'keyTimes="{KEY_TIMES_STR}" dur="{DUR}s" repeatCount="indefinite"/>'
        f'</circle>'
    )

# --- mean-field vector R̄ (arrow from center) -----------------------------
# length = R * DISC_R * 0.85; angle = psi
Rvec_x2 = [DISC_CX + Rk * DISC_R * 0.85 * math.cos(psi_k) for Rk, psi_k in R_keyframes]
Rvec_y2 = [DISC_CY + Rk * DISC_R * 0.85 * math.sin(psi_k) for Rk, psi_k in R_keyframes]

parts.append(
    f'<line x1="{DISC_CX}" y1="{DISC_CY}" stroke="{GOLD}" stroke-width="2.2" '
    f'stroke-linecap="round" opacity="0.95" filter="url(#soft-glow)">'
    f'<animate attributeName="x2" values="{smil_values(Rvec_x2)}" '
    f'keyTimes="{KEY_TIMES_STR}" dur="{DUR}s" repeatCount="indefinite" calcMode="spline" '
    f'keySplines="0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1"/>'
    f'<animate attributeName="y2" values="{smil_values(Rvec_y2)}" '
    f'keyTimes="{KEY_TIMES_STR}" dur="{DUR}s" repeatCount="indefinite" calcMode="spline" '
    f'keySplines="0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1"/>'
    f'</line>'
)
# tip of R vector — a small gold dot
parts.append(
    f'<circle r="3.6" fill="{GOLD}" filter="url(#soft-glow)">'
    f'<animate attributeName="cx" values="{smil_values(Rvec_x2)}" '
    f'keyTimes="{KEY_TIMES_STR}" dur="{DUR}s" repeatCount="indefinite" calcMode="spline" '
    f'keySplines="0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1"/>'
    f'<animate attributeName="cy" values="{smil_values(Rvec_y2)}" '
    f'keyTimes="{KEY_TIMES_STR}" dur="{DUR}s" repeatCount="indefinite" calcMode="spline" '
    f'keySplines="0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1;0.4 0 0.2 1"/>'
    f'</circle>'
)
# center dot
parts.append(f'<circle cx="{DISC_CX}" cy="{DISC_CY}" r="2.2" fill="{TEXT_DIM}"/>')

# disc label
parts.append(
    f'<text x="{DISC_CX}" y="{DISC_CY+DISC_R+46}" fill="{TEXT_DIM}" font-size="11" '
    f'text-anchor="middle" letter-spacing="2">PHASE SPACE · θ ∈ S¹</text>'
)
parts.append(
    f'<text x="{DISC_CX}" y="{DISC_CY+DISC_R+64}" fill="{TEXT}" font-size="12" '
    f'text-anchor="middle" font-style="italic">'
    f'R·e^{{iψ}} = (1/N) Σ e^{{iθₖ}}'
    f'</text>'
)

# --- MIDDLE divider ------------------------------------------------------
parts.append(
    f'<line x1="420" y1="60" x2="420" y2="{H-60}" stroke="{GRID}" stroke-width="1"/>'
)

# --- RIGHT: R(t) trace ---------------------------------------------------
# Axes
parts.append(
    f'<line x1="{TRACE_X0}" y1="{TRACE_Y1}" x2="{TRACE_X1}" y2="{TRACE_Y1}" '
    f'stroke="{GRID}" stroke-width="1"/>'
)
parts.append(
    f'<line x1="{TRACE_X0}" y1="{TRACE_Y0}" x2="{TRACE_X0}" y2="{TRACE_Y1}" '
    f'stroke="{GRID}" stroke-width="1"/>'
)

# y ticks 0, θ, 1
for y_val, label, colour in [(0.0, "0", TEXT_DIM), (THRESHOLD, "θ", GOLD), (1.0, "1", TEXT_DIM)]:
    y = TRACE_Y1 - (TRACE_Y1 - TRACE_Y0) * y_val
    parts.append(
        f'<line x1="{TRACE_X0-6}" y1="{y}" x2="{TRACE_X0}" y2="{y}" '
        f'stroke="{colour}" stroke-width="1"/>'
    )
    parts.append(
        f'<text x="{TRACE_X0-12}" y="{y+4}" fill="{colour}" font-size="11" '
        f'text-anchor="end">{label}</text>'
    )

# gate threshold horizontal line (dashed)
parts.append(
    f'<line x1="{TRACE_X0}" y1="{GATE_Y}" x2="{TRACE_X1}" y2="{GATE_Y}" '
    f'stroke="{GOLD}" stroke-width="1" stroke-dasharray="4 5" opacity="0.55"/>'
)
parts.append(
    f'<text x="{TRACE_X1-6}" y="{GATE_Y-6}" fill="{GOLD}" font-size="10.5" '
    f'text-anchor="end" letter-spacing="1" opacity="0.85">GATE · θ = 0.65</text>'
)

# trace path — densified via many keyframes: sample R(t) via cosine-smooth
# interpolation between R_keyframes (matches SMIL spline)
def smoothstep(a: float, b: float, t: float) -> float:
    t = max(0.0, min(1.0, t))
    s = t * t * (3 - 2 * t)
    return a + (b - a) * s

SAMPLES = 240
trace_pts: list[tuple[float, float]] = []
for s in range(SAMPLES + 1):
    t = s / SAMPLES  # 0..1
    # find segment
    for kk in range(len(KEY_TIMES) - 1):
        if KEY_TIMES[kk] <= t <= KEY_TIMES[kk + 1]:
            seg_t = (t - KEY_TIMES[kk]) / (KEY_TIMES[kk + 1] - KEY_TIMES[kk])
            R_s = smoothstep(R_keyframes[kk][0], R_keyframes[kk + 1][0], seg_t)
            break
    x = TRACE_X0 + (TRACE_X1 - TRACE_X0) * t
    y = TRACE_Y1 - (TRACE_Y1 - TRACE_Y0) * R_s
    trace_pts.append((x, y))

trace_d = "M " + " L ".join(f"{fmt(x)} {fmt(y)}" for x, y in trace_pts)
# static ghost trace (dim, full)
parts.append(
    f'<path d="{trace_d}" fill="none" stroke="{VIOLET}" stroke-width="1.2" opacity="0.25"/>'
)

# animated sweep: reveal path by animating stroke-dashoffset
# Compute approximate path length for dash animation
path_len = 0.0
for (x1, y1), (x2, y2) in pairwise(trace_pts):
    path_len += math.hypot(x2 - x1, y2 - y1)

parts.append(
    f'<path d="{trace_d}" fill="none" stroke="url(#trace-grad)" stroke-width="2.6" '
    f'stroke-linecap="round" filter="url(#soft-glow)" '
    f'stroke-dasharray="{fmt(path_len)}" stroke-dashoffset="{fmt(path_len)}">'
    f'<animate attributeName="stroke-dashoffset" values="{fmt(path_len)};0;{fmt(path_len*0.2)};0;{fmt(path_len)}" '
    f'keyTimes="{KEY_TIMES_STR}" dur="{DUR}s" repeatCount="indefinite" calcMode="linear"/>'
    f'</path>'
)

# moving marker on trace — animated circle along the same samples (using
# the sampled trace coordinates exactly, so marker = pen tip)
mx_vals = [p[0] for p in trace_pts]
my_vals = [p[1] for p in trace_pts]
# animate cx/cy with a dense values string (240 samples)
mx_str = ";".join(fmt(v) for v in mx_vals)
my_str = ";".join(fmt(v) for v in my_vals)
# pulsing gold dot on the R(t) tip
parts.append(
    f'<circle r="4.2" fill="{GOLD}" filter="url(#soft-glow)">'
    f'<animate attributeName="cx" values="{mx_str}" dur="{DUR}s" '
    f'repeatCount="indefinite" calcMode="linear"/>'
    f'<animate attributeName="cy" values="{my_str}" dur="{DUR}s" '
    f'repeatCount="indefinite" calcMode="linear"/>'
    f'<animate attributeName="r" values="3.4;5.2;3.6;5.2;3.4" '
    f'keyTimes="{KEY_TIMES_STR}" dur="{DUR}s" repeatCount="indefinite"/>'
    f'</circle>'
)

# state label (READY / BLOCKED) — synced with R crossing gate
# Determine per keyframe whether R > threshold
state_label_vals = []
state_color_vals = []
for Rk, _ in R_keyframes:
    if Rk >= THRESHOLD:
        state_label_vals.append("READY")
        state_color_vals.append(EMERALD)
    else:
        state_label_vals.append("BLOCKED")
        state_color_vals.append(VIOLET)

# Since SMIL <animate> on text content isn't widely supported, we stack
# two labels and animate their opacity in lockstep.
parts.append(
    f'<text x="{TRACE_X1}" y="{TRACE_Y0-14}" fill="{EMERALD}" font-size="13" '
    f'font-weight="700" text-anchor="end" letter-spacing="3">'
    f'READY'
    f'<animate attributeName="opacity" '
    f'values="{";".join("1" if s == "READY" else "0" for s in state_label_vals)}" '
    f'keyTimes="{KEY_TIMES_STR}" dur="{DUR}s" repeatCount="indefinite"/>'
    f'</text>'
)
parts.append(
    f'<text x="{TRACE_X1}" y="{TRACE_Y0-14}" fill="{VIOLET}" font-size="13" '
    f'font-weight="700" text-anchor="end" letter-spacing="3">'
    f'BLOCKED'
    f'<animate attributeName="opacity" '
    f'values="{";".join("1" if s == "BLOCKED" else "0" for s in state_label_vals)}" '
    f'keyTimes="{KEY_TIMES_STR}" dur="{DUR}s" repeatCount="indefinite"/>'
    f'</text>'
)

# x-axis label
parts.append(
    f'<text x="{(TRACE_X0+TRACE_X1)/2}" y="{TRACE_Y1+24}" fill="{TEXT_DIM}" font-size="11" '
    f'text-anchor="middle" letter-spacing="2">TIME · t</text>'
)
parts.append(
    f'<text x="{TRACE_X0-34}" y="{(TRACE_Y0+TRACE_Y1)/2}" fill="{TEXT_DIM}" font-size="11" '
    f'text-anchor="middle" letter-spacing="2" transform="rotate(-90 {TRACE_X0-34} {(TRACE_Y0+TRACE_Y1)/2})">'
    f'ORDER · R(t)</text>'
)

# footer tagline
parts.append(
    f'<text x="60" y="{H-16}" fill="{TEXT_DIM}" font-size="11" letter-spacing="2">'
    f'PHASE SYNCHRONIZATION AS EXECUTION GATE'
    f'</text>'
)
parts.append(
    f'<text x="{W-60}" y="{H-16}" fill="{TEXT_DIM}" font-size="11" text-anchor="end" '
    f'letter-spacing="2">COUPLED · BRAIN ⇌ MARKET</text>'
)

parts.append("</svg>")

out = Path(__file__).resolve().parents[1] / "docs" / "assets" / "hero_kuramoto.svg"
out.write_text("\n".join(parts), encoding="utf-8")
print(f"wrote {out} ({out.stat().st_size} bytes)")
