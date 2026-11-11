# Licence de-shadow / white-balance enhancement, ported from the web tool.
#
# Source of truth: ~/workspace/goals/qing-print-shop-id-document-tool/files/
#   证件一键处理-免安装版/证件一键处理.html
#   -> enhanceCanvas(src, LIC_ENH) with
#      LIC_ENH = { target: 248, gainCap: 2.2, black: 6 }   (ceil defaults to 255)
#   Comment there: cap 2.2 because on the hand-shadow sample a 2.8 gain field
#   scrambled the QR modules so they no longer decoded, while 2.2 decodes and
#   both samples needed gain <= 2.1, so de-shadowing is unharmed.
#
# The JS works on canvas RGBA pixels. This port works on OpenCV BGR uint8
# and converts to RGB float32 internally, so channel order matches the JS.
#
# Fidelity notes (read before "simplifying" this):
# - Background = downscale long edge to 420, then to 48, then upscale back
#   to full size. The 48px stage is a hard requirement: if the background
#   estimate picks up QR modules / strokes, division cancels local contrast
#   and washes the code out (JS comment: 105px stage failed zxing, 48px ok).
# - `ref` (per-channel 96th percentile of the background) is computed in the
#   JS but never used downstream -- gains use target/bgv directly. It is
#   computed here too and returned in info, so a future reader can see the
#   white-balance reference the JS derived; removing it changes nothing.
# - Red guard: strongly red pixels (seal / emblem) get G/B gains pulled
#   toward the R gain, otherwise per-channel division washes red to pink.
#   Second half: if the boosted pixel would clip at ceil, scale the whole
#   brightening increment back by one factor so channel gaps survive.
# - Ink guard: the darker a pixel is relative to its local background, the
#   less of the gain it eats (black QR modules keep only 22% of the gain),
#   so the gain field cannot scramble module contrast.
#
# Pipeline rule: this is for PHOTOS (shadowed paper). Flat electronic files
# / scans must bypass it -- on a clean render the division bleaches the
# warm paper tint to pure white (measured 2026-11-10: mean abs change 14.8,
# interior paper 13.7, gold frame band 31.6) for no benefit. See
# test_lic_enhance.py::test_electronic_near_identity and bigdoc.is_flat_scan.
import numpy as np
import cv2

LIC_ENH = {"target": 248, "gainCap": 2.2, "black": 6, "ceil": 255}


def _downscale_to(rgb, max_edge):
    """JS downscaleTo(): no-op copy if already small, else area/bicubic resize."""
    h, w = rgb.shape[:2]
    m = max(h, w)
    if m <= max_edge:
        return rgb.copy()
    sc = max_edge / m
    nw, nh = max(1, round(w * sc)), max(1, round(h * sc))
    return cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_AREA)


def _background_estimate(rgb):
    """Strictly low-frequency illumination field, JS enhanceCanvas method."""
    small = _downscale_to(rgb, 420)
    bg_small = _downscale_to(small, 48)
    h, w = rgb.shape[:2]
    return cv2.resize(bg_small, (w, h), interpolation=cv2.INTER_CUBIC)


def _paper_ref(bg):
    """Per-channel 96th percentile of the background, sampled every 4th pixel
    (JS steps i += 16 bytes over RGBA data). Returned for info only."""
    flat = bg[::4, ::4, :].reshape(-1, 3)
    ref = []
    for ch in range(3):
        vals = np.sort(flat[:, ch])
        idx = min(len(vals) - 1, int(np.ceil(0.96 * len(vals))) - 1)
        ref.append(float(max(60, vals[idx])) if len(vals) else 200.0)
    return ref


def enhance_license(bgr, opts=None, _bg=None):
    """De-shadow + white-balance a warped licence image (BGR uint8 in/out).

    opts: dict like LIC_ENH (target, gainCap, black, ceil). Defaults LIC_ENH.
    Returns (out_bgr_uint8, info) with gains / paper ref / per-pixel stats.
    Vectorized numpy port of JS enhanceCanvas; see test_lic_enhance.py for
    the scalar-reference equivalence check (max abs diff <= 1 on RGB).
    """
    o = dict(LIC_ENH)
    if opts:
        o.update(opts)
    if bgr is None or bgr.ndim != 3 or bgr.shape[2] != 3:
        raise ValueError("enhance_license: need a 3-channel BGR image")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32)
    bg = _bg if _bg is not None else _background_estimate(rgb)
    bg = bg.astype(np.float32)
    ref = _paper_ref(bg)

    target, gain_cap = float(o["target"]), float(o["gainCap"])
    black, ceil = float(o["black"]), float(o["ceil"])
    bgv = np.maximum(24.0, bg)
    gains = np.clip(target / bgv, 0.6, gain_cap)  # (H, W, 3), RGB order

    s = rgb
    # --- red guard, first half: pull G/B gains toward R gain for red pixels
    R = s[:, :, 0]
    red_rf = np.zeros(R.shape, np.float32)
    mask = R > 40
    if np.any(mask):
        pr = np.minimum((R - s[:, :, 1]) / np.maximum(R, 1e-6),
                        (R - s[:, :, 2]) / np.maximum(R, 1e-6))
        rf = np.clip((pr - 0.18) / 0.16, 0.0, 1.0)
        red_rf = np.where(mask, rf, 0.0).astype(np.float32)
    gains[:, :, 1] += (gains[:, :, 0] - gains[:, :, 1]) * red_rf
    gains[:, :, 2] += (gains[:, :, 0] - gains[:, :, 2]) * red_rf

    # --- ink guard: dark-vs-background pixels eat less of the gain
    r = s / bgv
    gf = np.where(r >= 0.85, 1.0,
                  np.where(r <= 0.55, 0.22, 0.22 + 0.78 * (r - 0.55) / 0.30))
    vs = s * (1.0 + (gains - 1.0) * gf)

    # --- red guard, second half: don't let R clip at ceil while G/B rise
    if np.any(red_rf > 0):
        t = np.ones(R.shape, np.float32)
        for ch in range(3):
            sv = s[:, :, ch]
            vv = vs[:, :, ch]
            cond = (vv > ceil) & (sv < ceil) & (vv > sv)
            cand = np.where(cond, (ceil - sv) / np.maximum(vv - sv, 1e-6), 1.0)
            t = np.minimum(t, cand.astype(np.float32))
        te = 1.0 - red_rf * (1.0 - t)
        vs = s + (vs - s) * te[:, :, None]

    out = np.clip((vs - black) * 255.0 / (ceil - black), 0, 255)
    out_bgr = cv2.cvtColor(out.astype(np.uint8), cv2.COLOR_RGB2BGR)
    info = {
        "target": target, "gainCap": gain_cap, "black": black, "ceil": ceil,
        "paper_ref_rgb": ref,
        "gain_mean_rgb": [float(gains[:, :, c].mean()) for c in range(3)],
        "gain_max_rgb": [float(gains[:, :, c].max()) for c in range(3)],
        "red_pixels_frac": float((red_rf > 0).mean()),
        "mean_abs_change": float(np.abs(out - s).mean()),
    }
    return out_bgr, info
