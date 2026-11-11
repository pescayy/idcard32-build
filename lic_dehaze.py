# Overexposure "washed-out" correction for licence photos (Qt line).
#
# Why this exists: LIC_ENH (lic_enhance.py) treats DARK / shadowed paper --
# it divides by the illumination field and lifts the paper toward 248. An
# overexposed glare shot has the opposite defect: the paper is already
# blown to white and the INK itself is lifted (black characters and QR
# modules photograph as light grey), so there is nothing for the divider
# to lift and the text stays pale. This stage runs AFTER LIC_ENH in
# lic_pipeline.process_license and only when the washout is measured,
# not guessed.
#
# v2 (2026-11-11, after ying rejected v1's look): v1 pressed the whole
# frame down with one luminance curve. The hue survived by construction
# (channels scaled together) but the DEPTH did not discriminate: the red
# seal and the emblem were dragged from pale pink (mean RGB ~ (233,111,
# 107)) to a loud full red, and the paper greyed -- "the colours look
# scary". v2 therefore discriminates by SATURATION and does less:
#
# 1. Saturation-protected correction. The correction curve is computed
#    exactly as before (luminance ratio, hue preserved), but the result
#    is blended per pixel with weight w(S) from the HSV saturation S:
#    w = 1 for S <= 0.20 (ink, paper, glare haze -- the things that are
#    actually washed), w = 0 for S >= 0.45 (red seal, emblem, gold
#    lettering -- saturated colour that must not be deepened), smooth
#    in between. Calibrated on the acceptance pack: seal pixels sit at
#    S p50 0.52-0.58, gold at 0.53, ink/paper below ~0.15.
# 2. Gentler dose, spent where it matters. Black point cap 110 -> 70
#    and the power 1.10 -> 1.05, but the blend weight also fades out
#    along a "white shoulder" (full weight at Y <= 190, zero at
#    Y >= 245): v1 greyed the paper because one curve pressed ink and
#    near-white paper alike, so v2 anchors the paper by construction
#    and spends the dose on the ink/haze band instead. The aim is
#    "ink deep enough, colours unstartling", erring on the side of
#    saving too little.
# 3. Stricter gate. The ink-floor threshold rises 70 -> 84. The pack's
#    floors form two clusters: genuinely blown shots at 96/99, and a
#    mild-glare burst at 73-78 whose colours are basically normal (v1
#    fired on them at bp 43-48). The 79..96 gap makes 84 a gate with
#    margin on both sides; below it this stage stays out entirely.
#    (A paper-haze-share criterion was measured and rejected: healthy
#    photos score 0.51-0.75 there, fully overlapping the washed class.)
#
# Trigger (all three must hold; measured on the post-LIC_ENH warped
# image, luminance Y = 0.299R + 0.587G + 0.114B):
#   ink_share  = fraction of pixels with Y < 80        <= 0.008
#   ink_floor  = 0.5th percentile of Y                 >= 84
#   clip_share = fraction of pixels with Y >= 250      >= 0.30
# Healthy/shadowed photos still clear with margin (ink floors <= 55 /
# <= 45, ink shares >= 0.0126 / >= 0.0174). Flat electronic files never
# reach this stage (pipeline skips it together with LIC_ENH), so their
# output is untouched by construction.
#
# Correction (only when triggered):
#   bp    = clamp(18 + (ink_floor - 84) * 2.6, 18, 70)
#   f(Y)  = 255 * ( clip((Y - bp) / (255 - bp), 0, 1) ) ** 1.05
# applied as a per-pixel luminance ratio to all three channels (hue
# preserved), then blended with the original by the product of the
# saturation weight above and the white-shoulder weight g(Y) (1 at
# Y <= 190, smooth to 0 at Y >= 245). Blown paper therefore stays
# white in fact, not just at the single point f(255) = 255; the
# measured ink floor of the worst sample (Y~114 at its darkest 1%)
# lands near Y~85 instead of v1's ~35.
import numpy as np
import cv2

INK_SHARE_MAX = 0.008
INK_FLOOR_MIN = 84.0
CLIP_SHARE_MIN = 0.30
GAMMA = 1.05
BP_BASE = 18.0
BP_SLOPE = 2.6
BP_MAX = 70.0
SAT_FULL = 0.20   # at/below this saturation the correction applies fully
SAT_ZERO = 0.45   # at/above this saturation the pixel is left alone
SHOULDER_LO = 190.0  # luminance up to here takes the full correction
SHOULDER_HI = 245.0  # luminance from here up is paper: left alone


def washout_stats(bgr):
    """Luminance statistics the trigger is defined on (BGR uint8 in)."""
    y = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ys = np.sort(y.ravel())
    n = len(ys)

    def pct(p):
        return float(ys[min(n - 1, int(p * n))])

    return {
        "ink_share": float((y < 80).mean()),
        "ink_floor_p005": pct(0.005),
        "ink_p01": pct(0.01),
        "ink_p02": pct(0.02),
        "clip_share": float((y >= 250).mean()),
        "paper_mean": float(y.mean()),
    }


def _saturation_weight(rgb):
    """Per-pixel blend weight from HSV saturation (float RGB in).

    Returns w in [0, 1]: 1 where the correction applies in full
    (desaturated ink / paper / haze), 0 where the pixel is saturated
    colour (seal, emblem, gold) and must be left as photographed.
    Smoothstep between SAT_FULL and SAT_ZERO so edges do not band.
    """
    mx = rgb.max(axis=2)
    mn = rgb.min(axis=2)
    s = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1.0), 0.0)
    t = np.clip((s - SAT_FULL) / (SAT_ZERO - SAT_FULL), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return 1.0 - smooth


def _shoulder_weight(y):
    """Blend weight along luminance: the white shoulder.

    1 in the ink/haze band (Y <= SHOULDER_LO), smoothstep down to 0 at
    paper white (Y >= SHOULDER_HI) so the correction can never grey
    the paper, however strong the black point is.
    """
    t = np.clip((y - SHOULDER_LO) / (SHOULDER_HI - SHOULDER_LO), 0.0, 1.0)
    smooth = t * t * (3.0 - 2.0 * t)
    return 1.0 - smooth


def dehaze_license(bgr):
    """Correct an overexposed washed-out warped licence image (v2).

    Returns (out_bgr_uint8, info). When the trigger does not fire the
    input is returned unchanged (same values, pixel for pixel) with
    info["triggered"] == False. info always carries the measured stats
    and, when triggered, the black point used, the saturation-weight
    summary, plus before/after paper mean and ink depth (2nd-percentile
    luminance).
    """
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("dehaze_license: need a 3-channel BGR image")
    st = washout_stats(bgr)
    info = {"triggered": False, **st}
    if not (st["ink_share"] <= INK_SHARE_MAX
            and st["ink_floor_p005"] >= INK_FLOOR_MIN
            and st["clip_share"] >= CLIP_SHARE_MIN):
        return bgr, info

    bp = float(min(BP_MAX, max(BP_BASE,
                               BP_BASE + (st["ink_floor_p005"]
                                          - INK_FLOOR_MIN) * BP_SLOPE)))
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    y = (0.299 * rgb[:, :, 0] + 0.587 * rgb[:, :, 1]
         + 0.114 * rgb[:, :, 2])
    t = np.clip((y - bp) / (255.0 - bp), 0.0, 1.0)
    y_new = 255.0 * np.power(t, GAMMA)
    ratio = y_new / np.maximum(y, 1.0)
    ratio = np.where(y >= 1.0, ratio, y_new)  # near-black: use as-is
    corrected = np.clip(rgb * ratio[:, :, None], 0, 255)

    sat_w = _saturation_weight(rgb)
    w = (sat_w * _shoulder_weight(y))[:, :, None]
    out = np.clip(rgb + w * (corrected - rgb), 0, 255).astype(np.uint8)
    out_bgr = cv2.cvtColor(out, cv2.COLOR_RGB2BGR)

    after = washout_stats(out_bgr)
    info.update({
        "triggered": True,
        "black_point": bp,
        "gamma": GAMMA,
        "sat_weight_mean": float(sat_w.mean()),
        "sat_protected_share": float((sat_w < 0.05).mean()),
        "blend_weight_mean": float(w.mean()),
        "paper_mean_after": after["paper_mean"],
        "ink_p02_after": after["ink_p02"],
    })
    return out_bgr, info
