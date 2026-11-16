# Licence (business licence, A4 landscape) processing glue for the Qt line.
# Pure function, no UI: app_qt.py calls process_license() and renders the
# result. The chain, after the 2026-11-16 dead-code purge:
#   quad detect (SCRFD -> contour -> contour on illumination-flattened)
#   -> two-frame formula paper quad (emblem-located; bigdoc
#      .expand_to_paper_edge when the emblem cannot be located; flat
#      scans short-circuit to the whole image)
#   -> ONE perspective warp to the output canvas
#   -> bigdoc.dark_border_fraction report.
# Zero-fabrication rule (2026-11-11 18:20): every output pixel must
# be a resampling of an original pixel; a warp quad poking outside
# the source is pulled back inside (geometry only) before the warp.
# Sideways sources (frame quad aspect clearly the transposed A4 ratio):
# the formula route already knows the document's top edge and warps
# document-ordered, so it comes out upright in a single run; only when
# the emblem cannot be located (search route) are both 90-degree
# rotations run through the pipeline and the better outcome kept
# (QR-in-document-bottom evidence, then no-fallback, then aspect fit,
# then cleanest border), with the returned paper_quad mapped back
# into the caller's original coordinates.
# Colour stages (lic_enhance + lic_dehaze) are kept in code but bypassed
# by default since 2026-11-11 (PS logic: automatic processing moves
# pixels, it does not change their values); pass tone=True to run them.
# Flat scans skip them either way (measured warm-paper bleaching).
# Dead-code purge (2026-11-16, ying's order): the TPS dewarp arm,
# the wedge crop/repaste/flat-white fill stage, the framing
# candidate arms race with its ink guards, converge and the
# corner/edge refinement stages are DELETED from this pipeline --
# retired with the strict two-frame scheme they served. The licence
# route is now: detect -> formula (or expansion) quad -> one warp.
# dewarp_skipped_high_bow stays in the result contract (always
# False) because the UI reads it.
# Degenerate-quad guard (same batch): detection candidates that are
# non-finite or zero-area are dropped at the source, a per-candidate
# expansion failure falls through to the next candidate, a converge
# failure keeps the unconverged quad, and process_license itself
# degrades any remaining per-image exception to the no_quad result --
# one bad photo must never crash the batch or the dialog (douyin
# n01/n08/n12 threw uncaught ValueError before this).
# A None from expand_to_paper_edge on every candidate quad (curled corner,
# heavy shadow) falls back to the best frame quad, flagged via
# info["expand_fallback"] so the UI can warn and offer manual marking.
import os

import numpy as np
import cv2

from bigdoc import expand_to_paper_edge, dark_border_fraction, \
    clamp_quad_to_frame, is_flat_scan, A4_RATIO
from lic_enhance import enhance_license
from lic_dehaze import dehaze_license

# Single output spec (2026-11-12 canvas ruling, measured at Qing's
# shop from her PS "image size" dialog): A4 landscape @200dpi. The
# old A4@300 canvas (3508x2480) and the A3 path (4961x3508) are gone:
# her licence output IS 2339x1654, shop A3 originals go to the flatbed
# scanner rather than the photo route, and the smaller canvas also
# caps the warp/TPS buffer sizes on her Win7-32/4GB machine. Detection
# still runs on the source photo at native resolution; this constant
# governs only the processing/output canvas.
OUT_A4 = (2339, 1654)  # A4 landscape: 297x210mm @200dpi

_BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(_BASE, "models", "carddetection_scrfd34gkps.onnx")


def order_quad(pts):
    p = np.asarray(pts, np.float64)
    s = p.sum(1)
    d = np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)],
                     p[np.argmax(s)], p[np.argmax(d)]])


def _quad_sane(quad, shape_hw=None):
    """Non-raising twin of bigdoc._validate_quad: True only for a finite
    4x2 quad whose corners are pairwise distinct, which does not cross
    itself (bowtie), and -- when the source shape is given -- whose true
    shoelace area and shortest edge clear floors set far below the
    smallest real licence in the gold set. Degenerate detector output
    (a zero-area SCRFD quad reached the pipeline in the 2026-11-11
    douyin batch) is filtered with this instead of being allowed to
    raise downstream.

    The 2026-11-14 tightening: the old test summed two cross products
    of corner-difference vectors, which is not a polygon area -- a
    quad carrying the same corner twice ([264,443] twice, the fault
    quad that produced a blank output with status ok) keeps both
    terms non-zero and walked straight through. Floors, from the
    31-sample gold set (frame quads: shortest edge 487.6 px, smallest
    area fraction 0.218 of frame): area >= 1% of frame and shortest
    edge >= 1% of frame diagonal sit >15x below the worst gold
    sample, so no real licence can trip them; degenerate output
    (repeated points, slivers, bowties) fails on every count."""
    try:
        q = np.asarray(quad, dtype=float)
    except Exception:
        return False
    if q.shape != (4, 2) or not np.all(np.isfinite(q)):
        return False
    # Pairwise distinctness: a repeated or near-repeated corner
    # collapses the quad regardless of scale.
    pair_d = [float(np.hypot(*(q[i] - q[j])))
              for i in range(4) for j in range(i + 1, 4)]
    if min(pair_d) < 2.0:
        return False
    # Self-intersection: opposite edges of an ordered quad must not
    # properly cross.
    def _orient(a, b, c):
        return (b[0] - a[0]) * (c[1] - a[1]) - \
            (b[1] - a[1]) * (c[0] - a[0])

    def _seg_cross(a, b, c, d):
        d1 = _orient(c, d, a)
        d2 = _orient(c, d, b)
        d3 = _orient(a, b, c)
        d4 = _orient(a, b, d)
        return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))
    if _seg_cross(q[0], q[1], q[2], q[3]) or \
            _seg_cross(q[1], q[2], q[3], q[0]):
        return False
    if shape_hw is not None:
        H, W = float(shape_hw[0]), float(shape_hw[1])
        area = 0.5 * abs(float(
            np.dot(q[:, 0], np.roll(q[:, 1], -1))
            - np.dot(q[:, 1], np.roll(q[:, 0], -1))))
        if area < 0.01 * H * W:
            return False
        edges = [float(np.hypot(*(q[i] - q[(i + 1) % 4])))
                 for i in range(4)]
        if min(edges) < 0.01 * float(np.hypot(W, H)):
            return False
    return True


def contour_detect(bgr):
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(gray, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    cnts = sorted(cnts, key=cv2.contourArea, reverse=True)
    for c in cnts[:5]:
        peri = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            return [tuple(map(float, p[0])) for p in approx]
    return None


def _flatten_illumination(bgr):
    """Divide by a heavily blurred background estimate to kill shadows."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    bg = cv2.GaussianBlur(gray, (0, 0), 60)
    flat = gray / (bg + 1e-3)
    flat = np.clip(flat * 200, 0, 255).astype(np.uint8)
    return cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)


def detect_license_quads(bgr, detector=None, info=None):
    """Candidate frame quads (4x2 TL,TR,BR,BL) in priority order, each with
    its route name. SCRFD first, then contour, then contour on the
    illumination-flattened image. May be empty.

    When a dict is passed as info, detection degradation is recorded
    on it instead of vanishing (2026-11-14 hardening: a bare except
    here once swallowed a detector broadcast error and the pipeline
    silently delivered a garbage contour output with status ok):
    info["scrfd_error"] holds the SCRFD exception text, and
    info["detect_degraded"] is set True whenever a detector was
    supplied but produced no usable SCRFD candidate (exception, miss,
    or a quad the sanity gate rejected) and the contour routes had to
    carry the photo. The return value is unchanged."""
    out = []
    scrfd_seen = False

    def _add(q, route):
        nonlocal scrfd_seen
        if q is None:
            return
        oq = order_quad(q)
        if _quad_sane(oq, bgr.shape[:2]):
            out.append((oq, route))
            if route == "scrfd":
                scrfd_seen = True

    if detector is not None:
        try:
            q = detector.detect(bgr)
        except Exception as exc:
            q = None
            if info is not None:
                info["scrfd_error"] = f"{type(exc).__name__}: {exc}"
        _add(q, "scrfd")
        if info is not None and not scrfd_seen:
            info["detect_degraded"] = True
    _add(contour_detect(bgr), "contour")
    _add(contour_detect(_flatten_illumination(bgr)), "contour_flat")
    return out


def detect_license_quad(bgr, detector=None):
    """Returns (best quad 4x2 TL,TR,BR,BL or None, route name or None)."""
    cands = detect_license_quads(bgr, detector)
    if not cands:
        return None, None
    return cands[0]


# ---------- auto-orientation for sideways sources ----------

def _transpose_evidence(quad):
    """Sideways test on the detected frame quad: its oriented aspect
    (top edge / left edge) must sit clearly on the transposed A4 ratio
    and clearly off the normal one. Anything ambiguous returns None --
    normal sources (flat scans included) are never rotated."""
    q = np.asarray(quad, float)
    w = float(np.hypot(*(q[1] - q[0])))
    h = float(np.hypot(*(q[3] - q[0])))
    if h <= 0:
        return None
    r = w / h
    dev_t = abs(r - 1.0 / A4_RATIO)
    dev_n = abs(r - A4_RATIO)
    # Windows calibrated on the measured packs (2026-11-11): genuinely
    # sideways sources read r = 0.55-0.68 (perspective eats up to ~0.16 of
    # the ideal 0.707), normal photo sources read r >= 1.34 -- the moat
    # between the families is wide, so the trigger demands r clearly
    # portrait (< 0.90) with the transposed fit inside 0.17 while the
    # normal fit is off by at least 0.45.
    if r < 0.90 and dev_t <= 0.17 and dev_n >= 0.45:
        return {"ratio_wh": round(r, 3),
                "dev_transposed": round(dev_t, 3),
                "dev_normal": round(dev_n, 3)}
    return None


def _qr_readable(warped):
    """True when a QR code is decodable in the warp. A readability signal
    only -- NOT a phase cue: the code's printed height varies by licence
    vintage (measured y=0.41 of page height on the 2026-11-11 real
    licence), so its position cannot separate upright from upside-down.
    """
    if warped is None:
        return False
    try:
        import zxingcpp
        return len(zxingcpp.read_barcodes(warped)) > 0
    except Exception:
        pass
    try:
        ok, _pts = cv2.QRCodeDetector().detectAndDecode(warped)
        return bool(ok)
    except Exception:
        return False


def _gold_upper_hit(warped):
    """Second orientation cue: the gold foil title + national emblem sit
    in the document's upper area. True when gold-hued bright pixels mass
    in the upper half, False when they mass in the lower, None when too
    few gold pixels to judge."""
    if warped is None:
        return None
    small = cv2.resize(warped, (640, 452), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
    gold = ((hsv[:, :, 0] >= 15) & (hsv[:, :, 0] <= 38) &
            (hsv[:, :, 1] >= 90) & (hsv[:, :, 2] >= 110))
    n = int(gold.sum())
    if n < 200:
        return None
    ys = np.nonzero(gold)[0]
    return bool(float(ys.mean()) < small.shape[0] * 0.5)


def _rotation_score(res):
    """Lexicographic outcome score for one rotated run; higher wins.
    The phase cue is the gold-mass-up test (the foil title + emblem are
    a layout invariant at the document's top); geometry (no fallback,
    aspect fit, clean border) and QR readability rank below it --
    geometry alone cannot tell an upright page from an upside-down one.
    """
    if res["status"] != "ok":
        return (-1, -1, 0, 0.0, 0.0)
    db = res["info"].get("dark_border", {})
    worst = max([db.get(k, 0.0) for k in ("top", "bottom", "left", "right")]
                or [0.0])
    return (1 if _gold_upper_hit(res["warped"]) else 0,
            1 if res["fallback"] is None else 0,
            1 if _qr_readable(res["warped"]) else 0,
            -abs(float(res["info"].get("aspect_dev", 9.0))),
            -worst)


def _unrotate_quad(quad, code, src_w, src_h):
    """Map a quad from rotated-image coordinates back to the original
    frame (src_w x src_h). cv2.rotate CW sends (x, y) to (src_h-1-y, x);
    CCW sends (x, y) to (y, src_w-1-x); these are the inverses."""
    q = np.asarray(quad, float)
    out = np.empty_like(q)
    if code == cv2.ROTATE_90_CLOCKWISE:
        out[:, 0] = q[:, 1]
        out[:, 1] = (src_h - 1) - q[:, 0]
    else:
        out[:, 0] = (src_w - 1) - q[:, 1]
        out[:, 1] = q[:, 0]
    return out


def _quad_outside_frame(shape_hw, quad):
    """True when any corner of a warp quad leaves the source image
    rectangle [0, W-1] x [0, H-1]. warpPerspective fills out-of-source
    samples with black -- fabricated pixels -- so under the
    zero-fabrication rule such a candidate is void, and a fallback
    quad is pulled back inside instead. For a convex quad the mapped
    sample set is the quad itself, so the corner test is exact; the
    regression pack additionally verifies every delivered warp on
    its full sampling grid."""
    H, W = int(shape_hw[0]), int(shape_hw[1])
    q = np.asarray(quad, dtype=np.float64).reshape(4, 2)
    return bool((q[:, 0] < 0.0).any() or (q[:, 0] > W - 1).any()
                or (q[:, 1] < 0.0).any() or (q[:, 1] > H - 1).any())




# ---------- two-frame formula expansion (2026-11-16, batch 1118) ----------
# The paper quad is COMPUTED from the detected frame quad, never searched
# for in the photo (searching was the old expand route's disease: shadows
# and desk seams pose as paper edges). Construction, calibrated on the
# 30-photo gold set (~/workspace/idcard-audit/license-twoframe-v2):
#   sheet dims are implied by the frame dims and the gap fractions
#   (whole-paper base -- pushing by frame-based amounts was the first
#   "red frame too small" bug); each DOCUMENT edge line of the frame
#   quad is offset outward by frac * sheet_dim and the new corners are
#   the intersections of adjacent offset lines (near-parallel edges
#   fall back to the averaged-corner push). Push amounts attach to
#   document edges, not photo edges: the document top edge (the one the
#   emblem rides on) always takes the top fraction, however the photo
#   is rotated -- attaching by photo edge cost 6/30 emblem escapes.
# Fractions: flat-subset medians (top 0.1379 / bot 0.06002 / sides
# 0.0437+0.04106), then two ying-approved adjustments, both re-measured
# on all 30: sides aligned to the larger value 0.0437 (corner median
# 22.2 -> 22.1, no gate change) and the top raised to 0.145 as emblem
# insurance (worst emblem margin 13.6 -> 22.1 px, corner median 26.4;
# the 0.152 variant cost corner 36.3 and was rejected).
PAPER_FRAC = {"top": 0.145, "right": 0.0437, "bot": 0.06002,
              "left": 0.0437}
_FQ_CORNERS = [(0, 1), (1, 2), (2, 3), (3, 0)]
_FQ_SIN_TOL = 0.02
# Emblem probe geometry on the 2048x1429 template canvas (frame rect
# (105,220)-(1951,1335); upright emblem centre (1035,223), measured).
_FQ_CANVAS = (2048, 1429)
_FQ_FRAME = np.array([[105, 220], [1951, 220], [1951, 1335],
                      [105, 1335]], np.float32)
_FQ_EMBLEM_ZONES = [(1035, 223), (1900, 780), (1021, 1332), (160, 770)]


def _fq_outward_normal(q, cen, ei):
    a, b = _FQ_CORNERS[ei]
    e = q[b] - q[a]
    n = np.array([-e[1], e[0]], float)
    n /= np.linalg.norm(n)
    mid = (q[a] + q[b]) / 2
    if np.dot(mid - cen, n) < 0:
        n = -n
    return n


def _fq_push_intersect(q, amounts):
    """Offset each edge line outward by amounts[ei]; corners are the
    intersections of adjacent offset lines. Near-parallel adjacent
    edges fall back to the averaged-corner push for that corner.
    Returns (quad, n_fallback_corners, min_abs_sin)."""
    cen = q.mean(axis=0)
    pts, dirs = [], []
    for ei in range(4):
        a, b = _FQ_CORNERS[ei]
        n = _fq_outward_normal(q, cen, ei)
        pts.append(q[a] + n * amounts[ei])
        dirs.append(q[b] - q[a])
    disp = [_fq_outward_normal(q, cen, ei) * amounts[ei]
            for ei in range(4)]
    avg = q.copy()
    for j in range(4):
        avg[j] = q[j] + (disp[(j - 1) % 4] + disp[j]) / 2
    out = q.copy()
    nfb, min_sin = 0, 1.0
    for j in range(4):
        e_prev, e_next = (j - 1) % 4, j
        d1, d2 = dirs[e_prev], dirs[e_next]
        cross = d1[0] * d2[1] - d1[1] * d2[0]
        sinv = abs(cross) / (np.linalg.norm(d1) * np.linalg.norm(d2))
        min_sin = min(min_sin, float(sinv))
        if sinv < _FQ_SIN_TOL:
            out[j] = avg[j]
            nfb += 1
            continue
        rhs = pts[e_next] - pts[e_prev]
        t = (rhs[0] * d2[1] - rhs[1] * d2[0]) / cross
        out[j] = pts[e_prev] + t * d1
    return out, nfb, min_sin


def _fq_docorder(geo, top_idx):
    return np.array([geo[(top_idx + i) % 4] for i in range(4)])


def _fq_geoorder(doc, top_idx):
    out = np.empty((4, 2))
    for i in range(4):
        out[(top_idx + i) % 4] = doc[i]
    return out


def emblem_top_edge_index(bgr, frame_quad):
    """Which geometric edge of the frame quad is the document's top
    edge (the one the emblem rides on): 0..3, or None when the emblem
    cannot be located. Warps the frame to the template canvas and
    looks for the red emblem ink near the four side zones -- the
    pipeline's own direction evidence (emblem position), the same
    measurement the gold calibration used."""
    q = np.asarray(frame_quad, np.float32)
    H = cv2.getPerspectiveTransform(q, _FQ_FRAME)
    w = cv2.warpPerspective(bgr, H, _FQ_CANVAS)
    b, g, r = cv2.split(w.astype(int))
    red = ((r - b > 25) & (r - g > 15) & (r > 100)).astype(np.uint8)
    n, _lab, stats, cent = cv2.connectedComponentsWithStats(red, 8)
    best = None
    for i in range(1, n):
        area = int(stats[i][4])
        if area <= 800:
            continue
        # The emblem is a compact disc (~244x268 on this canvas,
        # squeezed to ~400x150 when the document sits sideways in the
        # warp); elongated red background objects (folder edges, desk
        # items) and thin footer strips are not it, however large.
        bw, bh = int(stats[i][2]), int(stats[i][3])
        if not (90 <= bw <= 520 and 90 <= bh <= 520
                and 0.34 <= bw / bh <= 3.0):
            continue
        cx, cy = float(cent[i][0]), float(cent[i][1])
        for zi, (zx, zy) in enumerate(_FQ_EMBLEM_ZONES):
            dist = ((cx - zx) ** 2 + (cy - zy) ** 2) ** 0.5
            if dist <= 380 and (best is None or dist < best[0]):
                best = (dist, zi)
    return None if best is None else int(best[1])


def formula_paper_quad(bgr, frame_quad):
    """Two-frame formula: paper quad computed from the frame quad.
    Returns (paper_quad, info); paper_quad is None when the emblem
    cannot be located (caller falls back to the search route).
    Flat scans short-circuit exactly like expand_to_paper_edge."""
    q = np.asarray(frame_quad, float)
    if is_flat_scan(bgr, q):
        H, W = bgr.shape[:2]
        paper = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1],
                          [0, H - 1]], float)
        return paper, {"flat_scan": True, "paper_route": "formula",
                       "skipped_expansion": True}
    top_idx = emblem_top_edge_index(bgr, q)
    if top_idx is None:
        return None, {"paper_route": "formula_no_emblem"}
    doc = _fq_docorder(q, top_idx)
    doc_w = (np.linalg.norm(doc[1] - doc[0])
             + np.linalg.norm(doc[2] - doc[3])) / 2
    doc_h = (np.linalg.norm(doc[3] - doc[0])
             + np.linalg.norm(doc[2] - doc[1])) / 2
    sheet_w = doc_w / (1.0 - PAPER_FRAC["left"] - PAPER_FRAC["right"])
    sheet_h = doc_h / (1.0 - PAPER_FRAC["top"] - PAPER_FRAC["bot"])
    amounts = [PAPER_FRAC["top"] * sheet_h,
               PAPER_FRAC["right"] * sheet_w,
               PAPER_FRAC["bot"] * sheet_h,
               PAPER_FRAC["left"] * sheet_w]
    final_doc, nfb, msin = _fq_push_intersect(doc, amounts)
    return _fq_geoorder(final_doc, top_idx), {
        "paper_route": "formula", "formula_top_edge": int(top_idx),
        "formula_fallback_corners": int(nfb),
        "formula_min_sin": round(float(msin), 4)}


def _process_with_cands(bgr, cands, detector, out_size, tone):
    """The pipeline proper, given pre-detected candidate quads.

    Slim production shape (2026-11-16 dead-code purge, ying's
    yardstick reset): the first candidate whose paper quad reconciles
    -- two-frame formula first, search expansion as its fallback --
    is warped ONCE and shipped. The retired strict-scheme machinery
    that used to sit between the quad and the output (framing
    candidate arms race with ink guards, wedge crop/repaste/
    flat-white fill, TPS dewarp, converge, corner/edge refinement)
    is deleted, not gated: roughly-right framing plus the UI's
    manual corner marking is the settled product shape, and every
    deleted stage was measured CPU the licence route no longer pays.
    """
    # Try each candidate quad until one reconciles with the A4 paper edge.
    paper, info, route = None, {}, cands[0][1]
    frame_used = cands[0][0]
    expand_errors = []
    for quad, r in cands:
        try:
            # Two-frame formula first (2026-11-16): the paper quad is
            # computed from the frame quad. Only when the emblem cannot
            # be located does the candidate fall back to the old
            # search-the-photo expansion.
            p_try, i_try = formula_paper_quad(bgr, quad)
            if p_try is not None and i_try.get("paper_route") == "formula" \
                    and not i_try.get("flat_scan"):
                # The computed paper legitimately extends past the
                # photo's borders when the licence fills the frame
                # (the true paper edge is simply not in the photo).
                # Clip the corners into the image (geometry only, no
                # fabricated pixels) so the warp below samples only
                # real pixels.
                H0, W0 = bgr.shape[:2]
                q_clip = np.asarray(p_try, float).copy()
                q_clip[:, 0] = np.clip(q_clip[:, 0], 0, W0 - 1)
                q_clip[:, 1] = np.clip(q_clip[:, 1], 0, H0 - 1)
                if not np.allclose(q_clip, p_try):
                    i_try = dict(i_try)
                    i_try["formula_clipped"] = True
                    p_try = q_clip
            if p_try is None:
                p_try, i_try = expand_to_paper_edge(bgr, quad)
        except Exception as exc:
            # A degenerate candidate must not kill the photo: log it
            # and let the next candidate try (2026-11-11 douyin n08).
            expand_errors.append(f"{r}: {type(exc).__name__}: {exc}")
            continue
        info = i_try
        if p_try is not None and _quad_sane(p_try, bgr.shape[:2]):
            paper, route, frame_used = p_try, r, quad
            break
        if p_try is not None:
            expand_errors.append(f"{r}: expansion returned an "
                                 f"insane quad, ignored")
    if expand_errors:
        info = dict(info)
        info["expand_errors"] = expand_errors
    fallback = None
    if paper is None:
        # No candidate reconciles with A4 (curled corner, heavy shadow...).
        # bigdoc sanctions the frame quad as fallback: produce a usable
        # straightened output cropped at the printed frame, flagged so the
        # UI can warn and offer manual corner marking on top of it.
        paper = cands[0][0]
        fallback = "frame_quad"
        info = dict(info)
        info["expand_fallback"] = True

    info = dict(info)
    flat = bool(info.get("flat_scan"))
    formula_route = info.get("paper_route") == "formula"

    # Single warp for every route. Zero-fabrication guard: a quad
    # poking outside the source would print filled (black) pixels, so
    # its corners are pulled back inside (geometry only) first. The
    # formula route warps the document-ordered quad -- the formula
    # knows which geometric edge is the document's top -- so a
    # sideways or upside-down licence comes out upright in this one
    # run; the returned paper_quad stays geometric (the UI's
    # manual-marking contract).
    warp_quad = np.asarray(paper, float)
    if _quad_outside_frame(bgr.shape[:2], warp_quad):
        warp_quad, _ci = clamp_quad_to_frame(bgr.shape[:2],
                                             warp_quad, None)
        info["fallback_clamped"] = True
    src_quad = warp_quad
    if formula_route:
        _top = info.get("formula_top_edge")
        if _top in (1, 2, 3):
            src_quad = _fq_docorder(warp_quad, int(_top))
            info["formula_docorder_warp"] = True
    W, H = out_size
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                   np.float32)
    M = cv2.getPerspectiveTransform(src_quad.astype(np.float32), dst)
    warped = cv2.warpPerspective(bgr, M, (W, H))

    info["tone_enabled"] = bool(tone)
    info["enhanced"] = bool(tone) and not flat
    if tone and not flat:
        warped, einfo = enhance_license(warped)
        info["enhance_info"] = einfo
        # Overexposure washout stage: LIC_ENH lifts dark paper but cannot
        # deepen ink that glare has lifted to grey. dehaze_license only
        # acts when its measured trigger fires; otherwise it returns the
        # image pixel-identical (see lic_dehaze.py for the criterion).
        warped, dinfo = dehaze_license(warped)
        info["dehaze_info"] = dinfo
    info["dark_border"] = dark_border_fraction(warped)

    return {"status": "ok", "route": route, "flat": flat,
            "warped": warped, "paper_quad": warp_quad,
            "fallback": fallback,
            "dewarp_skipped_high_bow": False,
            "info": info}



def _process_license_impl(bgr, detector=None, out_size=OUT_A4,
                          model_path=None, tone=False, _depth=0):
    """Process one licence image. (Implementation; the public
    process_license wrapper below adds the per-image exception guard.)

    tone: explicit opt-in for the colour stages (lic_enhance shadow
    lifting + lic_dehaze washout correction). Default False -- geometry
    only, pixel values untouched (2026-11-11 PS-logic decision). Flat
    scans skip the colour stages regardless of this switch.

    Returns a dict:
      status: "ok" | "no_quad"
      route:  quad detection route that produced the paper quad
      flat:   True for electronic file / flat scan (whole image is paper)
      warped: BGR uint8 at out_size when status == "ok", else None
      paper_quad: converged paper quad in source coords (or None)
      fallback: None, or "frame_quad" when no candidate reconciled with A4
        and the printed frame itself was used as the paper (UI should warn
        and offer manual corner marking)
      dewarp_skipped_high_bow: always False since the 2026-11-16
        dead-code purge deleted the TPS arm; the key remains in the
        contract because the UI reads it
      info:   expand/converge/envelope/enhance/dark-border diagnostics;
        info["scrfd_error"] / info["detect_degraded"] are present when
        the SCRFD detector threw or produced no usable candidate and
        the contour routes carried the photo (2026-11-14 hardening);
        info["autorotate"] is present when a sideways source was
        auto-rotated (chosen direction + trigger evidence);
        (the TPS arm's info["dewarp"] verdict went with the purge)
    """
    if bgr is None or bgr.ndim != 3:
        raise ValueError("process_license: need a 3-channel BGR image")
    if detector is None:
        from core import CardDetectorAI
        detector = CardDetectorAI(model_path or DEFAULT_MODEL)

    det_info = {}
    cands = detect_license_quads(bgr, detector, det_info)

    def _with_det_info(res):
        """Merge the detection trace (scrfd_error / detect_degraded)
        under the result's own info, so a photo carried by the contour
        routes or the frame-quad fallback is visibly degraded in the
        result instead of looking like a clean SCRFD success."""
        if not det_info:
            return res
        info = dict(det_info)
        info.update(res["info"])
        res = dict(res)
        res["info"] = info
        return res

    if not cands:
        return _with_det_info(
            {"status": "no_quad", "route": None, "flat": False,
             "warped": None, "paper_quad": None, "fallback": None,
             "dewarp_skipped_high_bow": False,
             "info": {"reason": "no frame quad detected"}})

    if _depth == 0 and not is_flat_scan(bgr, cands[0][0]):
        trig = _transpose_evidence(cands[0][0])
        if trig is not None:
            # Slim first try (2026-11-16): the formula route knows
            # the document's top edge and warps document-ordered, so
            # a sideways licence comes out upright from ONE run of
            # the pipeline. Only when the formula cannot locate the
            # emblem (search route) do we pay for the
            # run-both-rotations comparison below.
            base0 = _process_with_cands(bgr, cands, detector,
                                        out_size, tone)
            if base0["status"] == "ok" and \
                    base0["info"].get("paper_route") == "formula":
                info = dict(base0["info"])
                info["autorotate"] = {"trigger": trig,
                                      "chosen": "formula_docorder"}
                base0["info"] = info
                return _with_det_info(base0)
            H, W = bgr.shape[:2]
            best = None
            scores = {}
            for name, code in (("cw", cv2.ROTATE_90_CLOCKWISE),
                               ("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE)):
                rr = process_license(cv2.rotate(bgr, code),
                                     detector=detector, out_size=out_size,
                                     tone=tone, _depth=1)
                sc = _rotation_score(rr)
                scores[name] = sc
                if best is None or sc > best[0]:
                    best = (sc, name, code, rr)
            if best is not None and best[3]["status"] == "ok":
                _sc, name, code, rr = best
                if rr["fallback"] is not None and \
                        not _qr_readable(rr["warped"]) and \
                        _gold_upper_hit(rr["warped"]) is None:
                    # The best rotation still only reached the frame
                    # fallback and carries no content evidence at all;
                    # if the original path reconciles with A4 on its own,
                    # keep it rather than trading geometry for a guess.
                    base = _process_with_cands(bgr, cands, detector,
                                               out_size, tone)
                    if base["fallback"] is None:
                        info = dict(base["info"])
                        info["autorotate"] = {
                            "trigger": trig, "chosen": "original",
                            "scores": {k: list(v) for k, v
                                       in scores.items()}}
                        base["info"] = info
                        return _with_det_info(base)
                rr = dict(rr)
                if rr["paper_quad"] is not None:
                    rr["paper_quad"] = _unrotate_quad(rr["paper_quad"],
                                                      code, W, H)
                info = dict(rr["info"])
                info["autorotate"] = {"trigger": trig, "chosen": name,
                                      "scores": {k: list(v) for k, v
                                                 in scores.items()}}
                rr["info"] = info
                return rr
            # both rotations failed: fall through to the original path

    return _with_det_info(_process_with_cands(bgr, cands, detector,
                                              out_size, tone))


def process_license(bgr, detector=None, out_size=OUT_A4, model_path=None,
                    tone=False, _depth=0):
    """Process one licence image; see _process_license_impl for the
    full result contract.

    Per-image exception guard (2026-11-11 batch): degenerate detector
    output on real-world photos (NaN / zero-area quads) used to throw
    uncaught ValueError out of converge/expand and kill the caller.
    Any unexpected per-image failure now degrades to the same no_quad
    result a detection miss produces -- the UI's existing manual
    corner-marking exit -- with the error recorded in
    info["error"]. A malformed argument (not a 3-channel image) still
    raises ValueError: that is a caller bug, not a photo outcome.
    """
    if bgr is None or bgr.ndim != 3:
        raise ValueError("process_license: need a 3-channel BGR image")
    try:
        return _process_license_impl(bgr, detector=detector,
                                     out_size=out_size,
                                     model_path=model_path, tone=tone,
                                     _depth=_depth)
    except Exception as exc:
        return {"status": "no_quad", "route": None, "flat": False,
                "warped": None, "paper_quad": None, "fallback": None,
                "dewarp_skipped_high_bow": False,
                "info": {"reason": "pipeline_error",
                         "error": f"{type(exc).__name__}: {exc}"}}
