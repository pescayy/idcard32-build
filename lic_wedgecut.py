# Crop + repaste + fill stage for the licence pipeline
# (ying's revised rule 2026-11-13; prototype verified on the 31-image
# gold set -- baseline four metrics zero drift, clutter total -27.9%,
# zero images worse; wired into lic_pipeline 2026-11-14, qt-perfect
# stage 3. Repaste is translation-only (no rescaling).
#
# 2026-11-14 third round (RESTORE-FILL): the SPEC-FIT round's wedge
# changes were put on trial by FINAL-CONFIG and the verdict split:
#   * Full-depth crops, spec-centre pinning and gradient fill are
#     REVERTED. The crop stage is again the baseline one (measured
#     depth, per-edge CROP_CAP_FRAC, geometric guard), and the
#     repaste again adjudicates per axis: the offset closest to the
#     spec-centre placement whose two fills both respect their 4%
#     caps -- an over-cap edge is clamped to its cap and the
#     remainder moves to the opposite edge (the 2026-11-14 per-edge
#     adjudication of WEDGE-PEREDGE, restored field-for-field
#     against wedge-peredge-rerun/run.json).
#   * Fill (DELIVERY LOCK, upper-level ruling 2026-11-14): flat
#     white 255, the baseline fill. This round's paper-extension
#     fill (colours sampled from the REAL paper strip just inside
#     the frame border, extended outward at the paper's own tone
#     -- see _fill_canvas_paper) is implemented but SWITCHED OFF
#     and archived in this module: its true paper tones were
#     billed in full by the clutter ruler (257,214 against the
#     130,526 baseline), the ruler stays unchanged, so the fill
#     yields and the flat baseline ships. Only fill-zone pixels
#     are ever written; the pasted paper is never touched.
#   * Kept from SPEC-FIT: the electronic standard rectangle
#     (SPEC_FRAME_RECT) stays in this module as the single standard
#     constant -- it still sources the joint fit's aspect ratio
#     (lic_edge_refine) and the margins are additionally reported
#     against it (info["std_margins"] / info["margin_diffs_std"]).
#     The placement spec the adjudication answers to is the
#     baseline one (the gold-set ruler reference below): the
#     per-edge baseline's outputs are the only fully passing
#     configuration on record, and restoring them pixel-for-pixel
#     outside the fill zones requires the baseline placement,
#     centre included (under the standard centre, shadow_top_half's
#     oy_spec drops 113 -> ~48 and the paste lands 45 px off its
#     baseline position). An over-cap fill DEMAND is still recorded
#     as info["fill_over_cap_warning"] (SPEC-FIT's field, kept):
#     the demand is a framing-suspicion signal worth surfacing,
#     even though placement itself is adjudicated per baseline.
#
# Mechanism (per edge, judged on its own evidence only):
#   1. On the pipeline's warped OUTPUT (before the TPS arm), measure
#      how deep off-paper clutter reaches inward along each edge --
#      the residual wedge a straight warp leaves where the real
#      sheet edge bows away from the sampling chord.
#   2. Crop that edge off at the deepest intrusion (capped per
#      edge). Good edges are not touched, and no edge borrows
#      another edge's line.
#   3. Paste the cropped image back onto the standard canvas
#      (2339x1654) positioned by the per-axis adjudication around
#      the spec-centre placement, and fill every gap flat white
#      (255). The pasted region is a verbatim copy (integer
#      offsets, no resampling): the content is never skewed or
#      stretched, and the frame can only move within the slack the
#      crops created on each axis.
#
# Fill red lines (never crossed):
#   * The crop line per edge is clamped by GEOMETRY, not by an
#     image-content guess: the caller maps the pipeline's own frame
#     quad through the warp homography, and the crop may never come
#     closer than (measured face-ink reach + GUARD_PX) to that frame
#     edge line. Filling therefore only ever covers the white-margin
#     zone outside the frame envelope; the frame and its interior
#     (emblem, text, QR) are never cropped away or fabricated.
#   * If the mapped frame edge leaves the canvas on an edge (frame
#     photographed missing / cut off by the warp), the stage
#     refuses and the image keeps the original pipeline output
#     (marked frame_missing) -- the existing error path.
#   * Per-edge fill is capped at FILL_CAP_FRAC of the opposing
#     axis' length by the adjudication itself (an over-cap demand
#     clamps to the cap, remainder to the opposite edge); per-edge
#     crop is capped at CROP_CAP_FRAC of that edge's own length,
#     deeper intrusion stays as a marked residual wedge.
#   * Flat scans never reach this stage (caller gates them out):
#     they are pixel-faithful originals.
#
# Why the guard is geometric: the frame band is pale guilloche, and
# output-space ink thresholds misread it both ways (a strict test
# skips the band and lands on interior text; a sensitive test trips
# on margin shadows). The pipeline already owns an exact frame
# estimate -- its quad plus the station-measured ink reach used by
# framing_ink_check / guard_quad_to_ink -- so the prototype reuses
# that geometry instead of inventing a second, weaker ruler. (The
# ink-threshold margin meter below is kept only as the evaluation's
# independent, admittedly biased, cross-check.)
#
# Spec position of the frame (placement): the gold-set ruler
# reference (license-goldset-v2.json ruler_reference -- a standalone
# electronic licence file, 2048x1429, frame outer rect
# L105 T220 R1951 B1335), scaled onto the output canvas. The ruler
# file is the high-top frame family; margin diffs against it are
# reported as measured, family geometry included, and diffs against
# the electronic standard are reported alongside (std keys above).
import numpy as np
import cv2

# Ruler reference: electronic licence file the gold set was ruled
# by. Placement spec of the restored baseline adjudication.
RULER_WH = (2048.0, 1429.0)
RULER_FRAME = (105.0, 220.0, 1951.0, 1335.0)  # L, T, R, B in ruler px

# Electronic standard: canvas and frame outer rect (L, T, R, B).
# Single standard constant of the pipeline (joint-fit aspect source
# in lic_edge_refine; margins additionally reported against it).
SPEC_CANVAS = (2339.0, 1654.0)
SPEC_FRAME_RECT = (118.0, 174.0, 2219.0, 1496.0)
SPEC_ASPECT = (SPEC_FRAME_RECT[2] - SPEC_FRAME_RECT[0]) / \
    (SPEC_FRAME_RECT[3] - SPEC_FRAME_RECT[1])

CROP_CAP_FRAC = 0.04    # per-edge crop cap, of the edge's own length
FILL_CAP_FRAC = 0.04    # per-edge fill cap, of the opposing axis
GUARD_PX = 2.0          # white kept between crop line and ink reach
MIN_DEPTH_PX = 2.0      # below this an edge counts as clean
EDGES = ("top", "right", "bottom", "left")

# Active fill (delivery lock, 2026-11-14): "flat255" ships. The
# "paper" paper-extension fill stays implemented below, switched
# off and archived (see the header note for the one-line reason).
FILL_MODE = "flat255"


def spec_margins(W, H):
    """Frame-envelope margins (L, T, R, B) of the placement spec
    (gold-set ruler reference), scaled onto a W x H canvas."""
    sx = float(W) / RULER_WH[0]
    sy = float(H) / RULER_WH[1]
    L = RULER_FRAME[0] * sx
    T = RULER_FRAME[1] * sy
    R = (RULER_WH[0] - RULER_FRAME[2]) * sx
    B = (RULER_WH[1] - RULER_FRAME[3]) * sy
    return L, T, R, B


def std_margins(W, H):
    """Frame-envelope margins (L, T, R, B) of the electronic
    standard (SPEC_FRAME_RECT), scaled onto a W x H canvas.
    Reporting only -- placement answers to spec_margins()."""
    sx = float(W) / SPEC_CANVAS[0]
    sy = float(H) / SPEC_CANVAS[1]
    L = SPEC_FRAME_RECT[0] * sx
    T = SPEC_FRAME_RECT[1] * sy
    R = (SPEC_CANVAS[0] - SPEC_FRAME_RECT[2]) * sx
    B = (SPEC_CANVAS[1] - SPEC_FRAME_RECT[3]) * sy
    return L, T, R, B


# ------------------------------------------------------------------
# Evaluation-only margin meter (ink thresholds; biased on pale frame
# bands, used identically on before/after outputs so the comparison
# is fair). NOT used by the stage itself.
# ------------------------------------------------------------------

def _level_raw(gray):
    H, W = gray.shape
    inner = gray[int(0.2 * H):int(0.8 * H), int(0.2 * W):int(0.8 * W)]
    return float(np.percentile(inner, 55))


def _first_ink_distances(mask_band, search):
    band = mask_band[:, :search]
    has = band.any(axis=1)
    if not has.any():
        return np.zeros(0), 0.0
    return band.argmax(axis=1)[has].astype(np.float64), \
        float(has.mean())


def measure_frame_margins(warped):
    """Distance from each canvas edge to the first strong ink,
    median over the central 60% of scan lines (strict 0.55 level /
    saturation 110 test, the pipeline family's convention). Returns
    (margins, found); a margin is None when fewer than half the
    scan lines find ink."""
    H, W = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)[:, :, 1] \
        .astype(np.float32)
    ink = (gray < 0.55 * _level_raw(gray)) | (sat >= 110)
    rows = slice(int(0.2 * H), int(0.8 * H), 2)
    cols = slice(int(0.2 * W), int(0.8 * W), 2)
    s_w = int(0.45 * W)
    s_h = int(0.45 * H)
    out = {}
    found = {}
    for name, band, s in (("left", ink[rows, :], s_w),
                          ("right", ink[rows, ::-1], s_w),
                          ("top", ink[:, cols].T, s_h),
                          ("bottom", ink[::-1, cols].T, s_h)):
        d, f = _first_ink_distances(band, s)
        out[name], found[name] = (float(np.median(d)), True) \
            if f >= 0.5 else (None, False)
    return out, found


# ------------------------------------------------------------------
# The stage.
# ------------------------------------------------------------------

def _intrusion_profile(stop_band):
    """Per scan line, the depth of the first run of two consecutive
    stop pixels (paper resumed, or vivid content reached). Lines
    with no such run inside the band read the full band depth and
    count as unresolved. Returns (depths, unresolved_count)."""
    both = stop_band[:, :-1] & stop_band[:, 1:]
    has = both.any(axis=1)
    depths = np.where(has, both.argmax(axis=1),
                      stop_band.shape[1]).astype(np.float64)
    return depths, int((~has).sum())


def _edge_line_distances(frame_out, W, H):
    """For each edge of the mapped frame quad: (min, mid) distance
    from the canvas edge to the frame edge line, sampled over the
    central 80% of the segment. frame_out is 4x2 TL,TR,BR,BL."""
    q = np.asarray(frame_out, dtype=np.float64).reshape(4, 2)
    ts = np.linspace(0.1, 0.9, 33)
    res = {}
    for i, e in enumerate(EDGES):
        a = q[i]
        b = q[(i + 1) % 4]
        pts = a[None, :] + ts[:, None] * (b - a)[None, :]
        if e == "top":
            dist = pts[:, 1]
        elif e == "bottom":
            dist = (H - 1) - pts[:, 1]
        elif e == "left":
            dist = pts[:, 0]
        else:
            dist = (W - 1) - pts[:, 0]
        res[e] = (float(dist.min()),
                  float(dist[len(dist) // 2]))
    return res


def _adjudicate(spec_off, slack, cap_lo, cap_hi):
    """Per-axis repaste adjudication (restored baseline): among the
    integer offsets whose two fills both respect their caps --
    offset in [max(0, slack - cap_hi), min(slack, cap_lo)] -- pick
    the one closest to the spec-centre offset. Returns None when no
    feasible offset exists (defensive; unreachable while every crop
    respects the same per-edge cap, since then slack <= cap_lo +
    cap_hi on each axis)."""
    lo = max(0.0, float(slack) - float(cap_hi))
    hi = min(float(slack), float(cap_lo))
    ilo = int(np.ceil(lo - 1e-9))
    ihi = int(np.floor(hi + 1e-9))
    if ilo > ihi:
        return None
    return min(max(int(spec_off), ilo), ihi)


def _edge_verdicts(crops, edet, fills_demanded, fills, fill_caps):
    """Per-edge adjudication record (restored baseline semantics).
    fills is None when no repaste happened (no_wedge): every edge
    then reads skipped with fill None. An edge whose spec-centre
    fill DEMAND exceeded its cap and was clamped back reads
    partial; an edge that was cropped and placed reads applied; an
    uncropped edge reads skipped with the reason the crop decision
    recorded for it (guard_blocked / clean / no_crop)."""
    out = {}
    for e in EDGES:
        reasons = []
        if fills is not None and fills_demanded[e] > fill_caps[e]:
            status = "partial"
            reasons.append("fill_clamped_to_cap")
        elif crops[e] > 0:
            status = "applied"
        else:
            status = "skipped"
            if "guard_blocked" in edet[e]["marks"]:
                reasons.append("guard_blocked")
            elif edet[e]["depth_max"] < MIN_DEPTH_PX:
                reasons.append("clean")
            else:
                reasons.append("no_crop")
        if "residual_wedge" in edet[e]["marks"]:
            reasons.append("residual_wedge")
        out[e] = {"status": status, "crop": int(crops[e]),
                  "fill": (None if fills is None
                           else int(fills[e])),
                  "fill_cap": round(float(fill_caps[e]), 1),
                  "reasons": reasons}
    return out


# ------------------------------------------------------------------
# Flat-white fill (ACTIVE -- FILL_MODE "flat255", the delivery
# configuration and the per-edge baseline's fill): every gap pixel
# is 255, the cropped paste sits verbatim at (ox, oy). Nothing
# else is written.
# ------------------------------------------------------------------

def _fill_canvas_flat(cropped, ox, oy, W, H):
    """Assemble the output canvas with a flat white (255) fill:
    the cropped paste at (ox, oy) verbatim, every other pixel
    255."""
    H2, W2 = cropped.shape[:2]
    canvas = np.full((H, W, 3), 255, np.uint8)
    canvas[oy:oy + H2, ox:ox + W2] = cropped
    return canvas


# ------------------------------------------------------------------
# Paper-extension fill (ARCHIVED -- not the active fill; FILL_MODE
# above governs and the delivery configuration fills flat white).
# The fill strips take their colours from the
# REAL paper strip just inside the frame border of the same edge --
# the sheet's own warm tone and its own gradient direction --
# extended outward across the fill zone. Per edge line (column for
# the top/bottom strips, row for the left/right ones) the sampler
# reads a strip of paper inside the frame (starting _PAPER_SKIP px
# past the frame line, _PAPER_DEPTH px deep), keeps paper-candidate
# pixels (saturation under the family's vivid ceiling -- print ink,
# the emblem and the QR code are not paper), and robustly estimates
# the colour at the frame line plus the tone slope toward it. The
# fill then continues that slope outward, damped, with the total
# excursion from the frame-line colour clamped small: the paper
# brightens or darkens gently toward its edge, it never leaps, and
# it is never clipped toward white to please a meter. Lines whose
# strip yields too few paper candidates borrow from their
# neighbours; an edge with almost no valid lines falls back to its
# own global paper median, and an edge with none at all (not seen
# on the gold set) stays flat white and says so in the info record.
# Only fill-zone pixels are written; the pasted paper is read-only.
# ------------------------------------------------------------------

_PAPER_SKIP = 6.0       # px inside the frame line before sampling
_PAPER_DEPTH = 34       # strip depth sampled inside the frame
_PAPER_NEAR_FRAC = 0.4  # near sub-band fraction of the strip
_PAPER_DAMP = 0.6       # outward slope damping
_PAPER_EXCURSION = 14.0  # max |deviation| from frame-line colour
_PAPER_SAT = 110        # paper-candidate vividness ceiling
_PAPER_MIN_PTS = 6      # candidate px per line for a valid sample
_PAPER_SMOOTH = 41      # cross-line median window
_PAPER_MIN_VALID = 0.2  # below this an edge uses its global median
# Smoothing round (2026-11-14, PAPERFILL-SMOOTH): the median window
# alone left the per-line anchor noise visible as faint vertical
# striping in the fill, each channel jittering on its own so the
# tone broke into cool and warm patches. After the median, the
# anchor profiles are Gaussian-smoothed along the line direction
# and then step-limited, so adjacent lines can only drift gently:
# the extension reads as one sheet of paper growing outward.
_PAPER_GAUSS = 8.0      # Gaussian sigma (lines) after the median
_PAPER_C0_STEP = 0.8    # max per-line change of frame-line colour
_PAPER_SLOPE_STEP = 0.04  # max per-line change of outward slope


def _median_filter_1d(a, k):
    """Median filter along axis 0 of an (N, C) array, edge-padded."""
    if k <= 1 or len(a) == 0:
        return a
    pad = k // 2
    ap = np.pad(a, ((pad, pad), (0, 0)), mode="edge")
    win = np.lib.stride_tricks.sliding_window_view(ap, k, axis=0)
    return np.median(win, axis=-1)


def _gaussian_filter_1d(a, sigma):
    """Gaussian smoothing along axis 0 of an (N, C) array,
    edge-padded. Used on the paper anchor profiles only."""
    if sigma <= 0 or len(a) < 3:
        return a
    r = max(1, int(3.0 * sigma))
    x = np.arange(-r, r + 1, dtype=np.float64)
    kern = np.exp(-0.5 * (x / sigma) ** 2)
    kern /= kern.sum()
    ap = np.pad(a, ((r, r), (0, 0)), mode="edge")
    out = np.empty_like(a)
    for ch in range(a.shape[1]):
        out[:, ch] = np.convolve(ap[:, ch], kern, mode="valid")
    return out


def _limit_steps_1d(a, max_step):
    """Limit adjacent-line steps of an (N, C) profile to max_step
    per channel, by repeated forward/backward clamping sweeps.
    Slow genuine gradients pass untouched; per-line jitter is
    ironed into a continuous drift."""
    if max_step <= 0 or len(a) < 2:
        return a
    a = a.copy()
    for _ in range(3):
        for i in range(1, len(a)):
            np.clip(a[i], a[i - 1] - max_step, a[i - 1] + max_step,
                    out=a[i])
        for i in range(len(a) - 2, -1, -1):
            np.clip(a[i], a[i + 1] - max_step, a[i + 1] + max_step,
                    out=a[i])
    return a


def _paper_profiles(samples, sats, valid_range):
    """Robust per-line paper colour at the frame line + outward
    slope, from one edge's inside-frame strip. samples is a
    (depth, lines, 3) array and sats/valid_range are (depth, lines)
    arrays sampled inward from the frame line (row 0 = closest to
    the line); valid_range marks
    in-range samples. Returns (c0, slope, valid_frac, mode) with
    c0/slope per line and channel."""
    D, N = sats.shape
    cand = valid_range & (sats < _PAPER_SAT)
    cnt = cand.sum(axis=0)
    ok = cnt >= _PAPER_MIN_PTS
    near_k = max(1, int(round(D * _PAPER_NEAR_FRAC)))
    far_k = max(near_k + 1, int(round(D * (1.0 - _PAPER_NEAR_FRAC))))
    s_pos = -(_PAPER_SKIP + np.arange(D, dtype=np.float64))
    s_near = float(s_pos[:near_k].mean())
    s_far = float(s_pos[far_k:].mean())

    def _band_med(lo_k, hi_k):
        sub = samples[lo_k:hi_k]
        subc = cand[lo_k:hi_k]
        out = np.full((N, 3), np.nan, np.float64)
        n = subc.sum(axis=0)
        good = n >= max(2, _PAPER_MIN_PTS // 2)
        if good.any():
            masked = np.where(subc[:, good][..., None],
                              sub[:, good], np.nan)
            out[good] = np.nanmedian(masked, axis=0)
        return out

    near = _band_med(0, near_k)
    far = _band_med(far_k, D)
    c0 = np.full((N, 3), np.nan, np.float64)
    slope = np.zeros((N, 3), np.float64)
    both = ~np.isnan(near[:, 0]) & ~np.isnan(far[:, 0])
    if both.any():
        slope[both] = (near[both] - far[both]) / (s_near - s_far)
        c0[both] = near[both] + slope[both] * (0.0 - s_near)
    only_near = ~np.isnan(near[:, 0]) & np.isnan(far[:, 0])
    c0[only_near] = near[only_near]
    valid = ~np.isnan(c0[:, 0])
    valid_frac = float(valid.mean()) if N else 0.0
    if valid_frac < _PAPER_MIN_VALID or not valid.any():
        if valid.any():
            gm = np.median(c0[valid], axis=0)
            gs = np.median(slope[valid], axis=0)
            return (np.repeat(gm[None], N, axis=0),
                    np.repeat(gs[None], N, axis=0),
                    valid_frac, "edge_global")
        return None, None, valid_frac, "flat255"
    idx = np.arange(N)
    for ch in range(3):
        c0[:, ch] = np.interp(idx, idx[valid], c0[valid, ch])
        slope[:, ch] = np.interp(idx, idx[valid], slope[valid, ch])
    c0 = _median_filter_1d(c0, _PAPER_SMOOTH)
    slope = _median_filter_1d(slope, _PAPER_SMOOTH)
    # Smoothing round: average out the residual per-line anchor
    # noise (striping / hue patches), then cap what is left.
    c0 = _gaussian_filter_1d(c0, _PAPER_GAUSS)
    slope = _gaussian_filter_1d(slope, _PAPER_GAUSS)
    c0 = _limit_steps_1d(c0, _PAPER_C0_STEP)
    slope = _limit_steps_1d(slope, _PAPER_SLOPE_STEP)
    return c0, slope, valid_frac, "per_line"


def _fill_canvas_paper(cropped, frame_c, ox, oy, W, H):
    """Assemble the output canvas: the cropped paste at (ox, oy)
    verbatim, every other pixel a paper extension from the inside
    of the frame (see the block comment above). frame_c is the
    frame quad in cropped-paste coordinates (4x2 TL,TR,BR,BL).
    Returns (canvas_uint8, fill_info)."""
    H2, W2 = cropped.shape[:2]
    canvas = np.full((H, W, 3), 255.0, np.float32)
    sat = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)[:, :, 1] \
        .astype(np.float32)
    img = cropped.astype(np.float32)
    D = _PAPER_DEPTH
    stats = {}

    def _line_interp(p, q, n, axis):
        """Frame-line coordinate per line along an edge from corner
        p to corner q (paste coords), for n lines."""
        t = (np.arange(n, dtype=np.float64) + 0.5)
        denom = (q[axis] - p[axis])
        if abs(denom) < 1e-9:
            return np.full(n, (p[1 - axis] + q[1 - axis]) / 2.0)
        tt = (t - p[axis]) / denom
        return p[1 - axis] + tt * (q[1 - axis] - p[1 - axis])

    def _strip(edge):
        """(samples, sats, valid_range, line_coord) for one edge,
        sampled inward from its frame line."""
        q = frame_c
        if edge == "top":
            lc = _line_interp(q[0], q[1], W2, 0)
            rows = np.round(lc)[None, :] + _PAPER_SKIP \
                + np.arange(D)[:, None]
            cols = np.arange(W2)[None, :]
            vr = (rows >= 0) & (rows < H2)
            ri = np.clip(rows, 0, H2 - 1).astype(int)
            ci = np.broadcast_to(cols, rows.shape).astype(int)
        elif edge == "bottom":
            lc = _line_interp(q[3], q[2], W2, 0)
            rows = np.round(lc)[None, :] - _PAPER_SKIP \
                - np.arange(D)[:, None]
            cols = np.arange(W2)[None, :]
            vr = (rows >= 0) & (rows < H2)
            ri = np.clip(rows, 0, H2 - 1).astype(int)
            ci = np.broadcast_to(cols, rows.shape).astype(int)
        elif edge == "left":
            lc = _line_interp(q[0], q[3], H2, 1)
            cols = np.round(lc)[None, :] + _PAPER_SKIP \
                + np.arange(D)[:, None]
            rows = np.arange(H2)[None, :]
            vr = (cols >= 0) & (cols < W2)
            ci = np.clip(cols, 0, W2 - 1).astype(int)
            ri = np.broadcast_to(rows, cols.shape).astype(int)
        else:  # right
            lc = _line_interp(q[1], q[2], H2, 1)
            cols = np.round(lc)[None, :] - _PAPER_SKIP \
                - np.arange(D)[:, None]
            rows = np.arange(H2)[None, :]
            vr = (cols >= 0) & (cols < W2)
            ci = np.clip(cols, 0, W2 - 1).astype(int)
            ri = np.broadcast_to(rows, cols.shape).astype(int)
        return img[ri, ci], sat[ri, ci], vr, lc

    def _render(edge, depth):
        """Colour one fill strip: returns an array shaped for direct
        placement (top/bottom: (depth, W2, 3) top-to-bottom;
        left/right: (H2, depth, 3) left-to-right), or None."""
        if depth <= 0:
            return None
        samples, sats, vr, lc = _strip(edge)
        c0, slope, vfrac, mode = _paper_profiles(samples, sats, vr)
        stats[edge] = {"valid_frac": round(vfrac, 3), "mode": mode,
                       "anchor_gray": (None if c0 is None else
                                       round(float(np.median(
                                           c0.mean(axis=1))), 1))}
        if c0 is None:
            return None
        if edge == "top":
            # canvas rows 0..oy-1; outward distance s = frame line
            # canvas coordinate minus the canvas row.
            rows_canvas = (oy - depth) + np.arange(depth)[:, None]
            s = (oy + lc)[None, :] - rows_canvas
            vals = c0[None] + np.clip(
                _PAPER_DAMP * slope[None] * s[..., None],
                -_PAPER_EXCURSION, _PAPER_EXCURSION)
            return vals
        if edge == "bottom":
            rows_canvas = (oy + H2) + np.arange(depth)[:, None]
            s = rows_canvas - (oy + lc)[None, :]
            vals = c0[None] + np.clip(
                _PAPER_DAMP * slope[None] * s[..., None],
                -_PAPER_EXCURSION, _PAPER_EXCURSION)
            return vals
        if edge == "left":
            cols_canvas = (ox - depth) + np.arange(depth)[None, :]
            s = (ox + lc)[:, None] - cols_canvas
            vals = c0[:, None] + np.clip(
                _PAPER_DAMP * slope[:, None] * s[..., None],
                -_PAPER_EXCURSION, _PAPER_EXCURSION)
            return vals
        cols_canvas = (ox + W2) + np.arange(depth)[None, :]
        s = cols_canvas - (ox + lc)[:, None]
        vals = c0[:, None] + np.clip(
            _PAPER_DAMP * slope[:, None] * s[..., None],
            -_PAPER_EXCURSION, _PAPER_EXCURSION)
        return vals

    placed = {}
    top = _render("top", oy)
    if top is not None:
        placed["top"] = top
        canvas[0:oy, ox:ox + W2] = top
    bot = _render("bottom", H - (oy + H2))
    if bot is not None:
        placed["bottom"] = bot
        canvas[oy + H2:H, ox:ox + W2] = bot
    left = _render("left", ox)
    if left is not None:
        placed["left"] = left
        canvas[oy:oy + H2, 0:ox] = left
    right = _render("right", W - (ox + W2))
    if right is not None:
        placed["right"] = right
        canvas[oy:oy + H2, ox + W2:W] = right
    # Corners: the average of the two adjacent strips' extensions
    # (each strip's nearest line replicated across the corner).
    if oy > 0 and ox > 0 and "top" in placed and "left" in placed:
        v = np.repeat(placed["top"][:, 0:1], ox, axis=1)
        hz = np.repeat(placed["left"][0:1], oy, axis=0)
        canvas[0:oy, 0:ox] = 0.5 * (v + hz)
    if oy > 0 and W - (ox + W2) > 0 and "top" in placed \
            and "right" in placed:
        v = np.repeat(placed["top"][:, -1:], W - (ox + W2), axis=1)
        hz = np.repeat(placed["right"][0:1], oy, axis=0)
        canvas[0:oy, ox + W2:W] = 0.5 * (v + hz)
    if H - (oy + H2) > 0 and ox > 0 and "bottom" in placed \
            and "left" in placed:
        v = np.repeat(placed["bottom"][:, 0:1], ox, axis=1)
        hz = np.repeat(placed["left"][-1:], H - (oy + H2), axis=0)
        canvas[oy + H2:H, 0:ox] = 0.5 * (v + hz)
    if H - (oy + H2) > 0 and W - (ox + W2) > 0 \
            and "bottom" in placed and "right" in placed:
        v = np.repeat(placed["bottom"][:, -1:], W - (ox + W2),
                      axis=1)
        hz = np.repeat(placed["right"][-1:], H - (oy + H2), axis=0)
        canvas[oy + H2:H, ox + W2:W] = 0.5 * (v + hz)
    canvas[oy:oy + H2, ox:ox + W2] = img
    return np.clip(canvas, 0, 255).astype(np.uint8), stats


def crop_repave_fill(warped, frame_out, ink_reach=None):
    """Apply the crop + repaste + flat-white-fill stage to one
    warped output (pre-TPS). frame_out: the pipeline's frame quad mapped
    through the warp homography into canvas coordinates (4x2).
    ink_reach: per-edge maximum face-ink reach beyond the frame
    line (source-side station measurement, robustified), in canvas
    px -- pass the caller's values rescaled by the warp's local
    scale, or None for zeros. Returns (image, info); on any refusal
    the ORIGINAL image comes back with applied=False and the reason
    marked."""
    H, W = warped.shape[:2]
    spec = spec_margins(W, H)
    spec_d = dict(zip(("left", "top", "right", "bottom"), spec))
    std = std_margins(W, H)
    std_d = dict(zip(("left", "top", "right", "bottom"), std))
    info = {"mechanism": "crop_repave_whitefill_v2",
            "spec_margins": {k: round(v, 1)
                             for k, v in spec_d.items()},
            "std_margins": {k: round(v, 1)
                            for k, v in std_d.items()}}
    if frame_out is None or not np.all(np.isfinite(
            np.asarray(frame_out, dtype=np.float64))):
        return warped, dict(info, applied=False, accepted=False,
                            refused="no_frame_geometry")
    frame_out = np.asarray(frame_out, dtype=np.float64).reshape(4, 2)
    reach = {e: 0.0 for e in EDGES}
    if ink_reach is not None:
        for e, v in zip(EDGES, ink_reach):
            reach[e] = float(v)
    dists = _edge_line_distances(frame_out, W, H)
    info["frame_min_dist"] = {e: round(dists[e][0], 1) for e in EDGES}
    info["frame_mid_dist"] = {e: round(dists[e][1], 1) for e in EDGES}
    info["ink_reach"] = {e: round(reach[e], 1) for e in EDGES}
    missing = [e for e in EDGES if dists[e][0] <= 0.0]
    if missing:
        return warped, dict(info, applied=False, accepted=False,
                            refused="frame_missing", missing=missing)

    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)[:, :, 1] \
        .astype(np.float32)
    paper_level = 0.82 * _level_raw(gray)
    info["paper_level"] = round(float(paper_level), 1)
    stop = (gray >= paper_level) | (sat >= 110)

    # --- per-edge intrusion + crop decision (baseline) ------------
    lim_tb = int(0.18 * H)
    lim_lr = int(0.18 * W)
    cols = slice(int(0.1 * W), int(0.9 * W), 2)
    rows = slice(int(0.1 * H), int(0.9 * H), 2)
    profiles = {}
    dep, unr = _intrusion_profile(stop[:lim_tb, cols].T)
    profiles["top"] = (dep, unr)
    dep, unr = _intrusion_profile(stop[::-1, cols].T[:lim_tb])
    profiles["bottom"] = (dep, unr)
    dep, unr = _intrusion_profile(stop[rows, :lim_lr])
    profiles["left"] = (dep, unr)
    dep, unr = _intrusion_profile(stop[rows, ::-1][:, :lim_lr])
    profiles["right"] = (dep, unr)
    edge_len = {"top": float(W), "bottom": float(W),
                "left": float(H), "right": float(H)}
    fill_caps = {"left": FILL_CAP_FRAC * H,
                 "right": FILL_CAP_FRAC * H,
                 "top": FILL_CAP_FRAC * W,
                 "bottom": FILL_CAP_FRAC * W}
    crops = {}
    edet = {}
    for e in EDGES:
        dep, unr = profiles[e]
        depth_max = float(dep.max()) if len(dep) else 0.0
        cap = CROP_CAP_FRAC * edge_len[e]
        guard_limit = dists[e][0] - reach[e] - GUARD_PX
        c = 0
        marks = []
        if depth_max >= MIN_DEPTH_PX:
            c = int(min(depth_max, cap, guard_limit))
            if c < MIN_DEPTH_PX:
                c = 0
                marks.append("guard_blocked")
            elif depth_max > c:
                marks.append("residual_wedge")
        crops[e] = c
        edet[e] = {"depth_max": round(depth_max, 1),
                   "unresolved_lines": int(unr),
                   "crop_cap": round(cap, 1),
                   "guard_limit": round(guard_limit, 1),
                   "crop": c, "marks": marks}
    info["edges"] = edet
    info["crops"] = dict(crops)
    if all(v == 0 for v in crops.values()):
        info["edge_verdicts"] = _edge_verdicts(
            crops, edet, None, None, fill_caps)
        return warped, dict(info, applied=False, accepted=False,
                            reason="no_wedge")

    cT, cR, cB, cL = crops["top"], crops["right"], crops["bottom"], \
        crops["left"]
    cropped = warped[cT:H - cB, cL:W - cR]
    H2, W2 = cropped.shape[:2]
    if W2 <= 0 or H2 <= 0:
        return warped, dict(info, applied=False, accepted=False,
                            refused="crop_consumed_canvas")

    # --- repaste: spec-centre placement, then per-axis ------------
    # adjudication back inside the fill caps (restored baseline).
    cen = frame_out.mean(axis=0)
    cx2 = float(cen[0]) - cL
    cy2 = float(cen[1]) - cT
    scx = (spec_d["left"] + (W - spec_d["right"])) / 2.0
    scy = (spec_d["top"] + (H - spec_d["bottom"])) / 2.0
    ox0 = int(round(scx - cx2))
    oy0 = int(round(scy - cy2))
    ox_c = min(max(ox0, 0), W - W2)
    oy_c = min(max(oy0, 0), H - H2)
    clamped = (ox_c != ox0) or (oy_c != oy0)
    fills_demanded = {"left": ox_c, "right": W - (ox_c + W2),
                      "top": oy_c, "bottom": H - (oy_c + H2)}
    info["fills_demanded"] = dict(fills_demanded)
    info["fill_caps"] = {k: round(v, 1) for k, v in fill_caps.items()}
    info["offsets_spec_centre"] = [ox_c, oy_c]
    info["offset_clamped"] = bool(clamped)
    over = [e for e in fills_demanded
            if fills_demanded[e] > fill_caps[e]]
    if over:
        info["fill_over_cap_warning"] = {
            e: {"fill": int(fills_demanded[e]),
                "cap": round(float(fill_caps[e]), 1)} for e in over}
    ox = _adjudicate(ox_c, W - W2, fill_caps["left"],
                     fill_caps["right"])
    oy = _adjudicate(oy_c, H - H2, fill_caps["top"],
                     fill_caps["bottom"])
    if ox is None or oy is None:
        # Defensive only (see _adjudicate): keep the baseline
        # refusal shape rather than inventing a placement.
        info["fills"] = dict(fills_demanded)
        info["offsets"] = [ox_c, oy_c]
        info["edge_verdicts"] = _edge_verdicts(
            crops, edet, fills_demanded, fills_demanded, fill_caps)
        return warped, dict(info, applied=False, accepted=False,
                            refused="fill_over_cap", over_cap=over)
    fills = {"left": ox, "right": W - (ox + W2),
             "top": oy, "bottom": H - (oy + H2)}
    info["fills"] = dict(fills)
    info["offsets"] = [ox, oy]
    info["offset_adjusted"] = bool([ox, oy] != [ox_c, oy_c])
    info["edge_verdicts"] = _edge_verdicts(crops, edet,
                                           fills_demanded, fills,
                                           fill_caps)

    if FILL_MODE == "paper":
        # Archived paper-extension fill (switched off, see header).
        info["mechanism"] = "crop_repave_paperfill_v5"
        frame_c = frame_out - np.array([cL, cT], dtype=np.float64)
        canvas, fill_stats = _fill_canvas_paper(cropped, frame_c,
                                                ox, oy, W, H)
        info["fill"] = {"mode": "paper_extension",
                        "edges": fill_stats}
    else:
        canvas = _fill_canvas_flat(cropped, ox, oy, W, H)
        info["fill"] = {"mode": "flat255"}
    m_after = {"left": dists["left"][1] - cL + ox,
               "right": dists["right"][1] - cR + (W - (ox + W2)),
               "top": dists["top"][1] - cT + oy,
               "bottom": dists["bottom"][1] - cB + (H - (oy + H2))}
    info["margins_after"] = {k: round(v, 1)
                             for k, v in m_after.items()}
    info["margin_diffs"] = {
        k: round(m_after[k] - spec_d[k], 1) for k in m_after}
    info["margin_diffs_std"] = {
        k: round(m_after[k] - std_d[k], 1) for k in m_after}
    return canvas, dict(info, applied=True, accepted=True)
