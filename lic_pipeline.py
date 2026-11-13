# Licence (business licence, A4/A3 landscape) processing glue for the Qt line.
# Pure function, no UI: app_qt.py calls process_license() and renders the
# result. Chains the modules that are already tested in isolation:
#   quad detect (SCRFD -> contour -> contour on illumination-flattened)
#   -> bigdoc.expand_to_paper_edge (flat scans short-circuit to whole image)
#   -> bigdoc.converge_corners -> perspective warp to output size
#   -> bigdoc.dark_border_fraction report.
# Post-converge framing (2026-11-11, final calibration): three
# candidate framings are warped and scored on their actual outputs
# (QR readable first, then cleanest dark border; near-ties prefer
# the straighter framing) -- no candidate wins by construction:
#   expand   -- the reconciled paper quad as-is;
#   tighten  -- expand quad + bigdoc.clamp_quad_to_frame (edges
#     poking outside the source pulled back in; they would print as
#     black bands) + bigdoc.tighten_edges (edges swallowing
#     background along a curled sheet edge shrunk past the measured
#     intrusion), Qing's lose-a-sliver rule;
#   envelope -- bigdoc.envelope_quad: the straight-line envelope of
#     the printed frame -- the frame lands as a perfect rectangle
#     with a thin, even line of paper white around it (margins equal
#     in pairs, constant along each edge), never following the
#     paper's curled outline, never sampling outside the paper and
#     never cutting the frame's bowed ink or the face content that
#     rides on it.
# All of these only move geometry, never pixel values, and never
# cross the printed frame.
# Zero-fabrication rule (2026-11-11 18:20): every output pixel must
# be a resampling of an original pixel. There is no padding or
# borrowed-paper path anywhere in the framing layer (the padded
# envelope of the previous calibration was rejected on sight), and
# any candidate whose warp would sample outside the source image is
# void, however clean its border would otherwise look. Symmetric
# margins are the target on a normally-shot flat sheet; when the
# photo itself is skewed or curled and one side's real paper runs
# short, the framing takes what reality offers -- asymmetric is
# acceptable, invented is not.
# Sideways sources (frame quad aspect clearly the transposed A4 ratio)
# are auto-rotated: both 90-degree rotations are run through the full
# pipeline and the better outcome wins (QR-in-document-bottom evidence,
# then no-fallback, then aspect fit, then cleanest border); the returned
# paper_quad is mapped back into the caller's original coordinates.
# Colour stages (lic_enhance + lic_dehaze) are kept in code but bypassed
# by default since 2026-11-11 (PS logic: automatic processing moves
# pixels, it does not change their values); pass tone=True to run them.
# Flat scans skip them either way (measured warm-paper bleaching).
# TPS dewarp (lic_undistort, wired 2026-11-11 after the 15-photo +
# 30-pack evidence rounds): runs on the photo route after the warp,
# before toning. Flat scans are hard-gated out (pixel-faithful
# originals; the arm's ruler reads phantom bow on electronic ornament).
# The arm is self-verifying -- unless it provably reduces the measured
# bow it returns the warp pixel-identical, so untriggered and
# rolled-back outputs are byte-equal to the pre-wiring pipeline. A
# measured bow above the gate that the self-check refuses to touch is
# flagged on the result (dewarp_skipped_high_bow) for the UI to report
# honestly.
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

from bigdoc import expand_to_paper_edge, converge_corners, \
    dark_border_fraction, clamp_quad_to_frame, tighten_edges, \
    envelope_quad, is_flat_scan, A4_RATIO, refine_corners_local, \
    edge_station_bounds, framing_ink_check, guard_quad_to_ink
from bigdoc import _robustify_bounds, _edge_geometry
from lic_enhance import enhance_license
from lic_dehaze import dehaze_license
from lic_wedgecut import crop_repave_fill
from lic_edge_refine import refine_frame_edges
from lic_undistort import undistort_license, gate_max_dev_px

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


# Delivery configuration for the framing-refinement stages
# (FINAL-CONFIG, 2026-11-14): both stages OFF in the shipped
# pipeline. Neither refinement beat the no-refinement production
# baseline on the 31-image gold set -- per-edge rebuild (v1):
# corner mean 43.2 vs baseline 42.7; joint rectangle fit
# (SPEC-FIT): 47.2 and impurity 256,065 vs baseline 130,526 --
# because the internal evidence cost is not a reliable proxy for
# the gold truth and the downstream framing vote is non-monotone
# in quad quality (SPEC-FIT-RUN.md SS1.4). The code stays in
# lic_edge_refine for a future round; these constants are the
# single switch point. The rule-slimming changes of the same
# round (single spec source in lic_wedgecut, full-depth wedge
# cut, fill_over_cap_warning, gradient fill, wedge QR guard)
# are NOT gated by these switches and stay in effect.
EDGE_REFINE_EDGE_STAGE = False
EDGE_REFINE_JOINT_STAGE = False


def _process_with_cands(bgr, cands, detector, out_size, tone):
    """The pipeline proper, given pre-detected candidate quads."""
    # Try each candidate quad until one reconciles with the A4 paper edge.
    paper, info, route = None, {}, cands[0][1]
    frame_used = cands[0][0]
    expand_errors = []
    # Edge refinement (2026-11-14, lic_edge_refine): a detector quad's
    # corners are point estimates -- one shadow-dragged corner tilts
    # its whole edge line, and the slant survives every later stage
    # because they all trust the frame. Re-measure each SCRFD edge
    # along its full length on the original pixels and rebuild the
    # corners as line intersections before the paper expansion
    # consumes the quad; edges without trustworthy evidence keep the
    # detector line, flat scans (corners exact by construction) skip
    # the stage, and any failure leaves the candidate untouched.
    # DELIVERY CONFIG: both stages are switched OFF (see the
    # EDGE_REFINE_* constants above) -- the call below then returns
    # every candidate untouched and the detector quads flow through
    # exactly as in the HARDEN+WEDGE production baseline; each
    # candidate's info still records the disabled state.
    edge_refine_infos = []
    refined_cands = []
    for quad, r in cands:
        if r != "scrfd":
            refined_cands.append((quad, r))
            continue
        try:
            if is_flat_scan(bgr, quad):
                edge_refine_infos.append(
                    {"route": r, "applied": False,
                     "reason": "flat_scan_gate"})
                refined_cands.append((quad, r))
                continue
            q2, rinfo = refine_frame_edges(
                bgr, quad,
                edge_stage=EDGE_REFINE_EDGE_STAGE,
                joint_stage=EDGE_REFINE_JOINT_STAGE)
            if rinfo.get("applied") and not _quad_sane(q2, bgr.shape[:2]):
                rinfo = dict(rinfo)
                rinfo["applied"] = False
                rinfo["reason"] = "refined_insane"
                q2 = quad
        except Exception as exc:
            edge_refine_infos.append(
                {"route": r, "applied": False,
                 "error": f"{type(exc).__name__}: {exc}"})
            refined_cands.append((quad, r))
            continue
        rinfo = dict(rinfo)
        rinfo["route"] = r
        edge_refine_infos.append(rinfo)
        refined_cands.append((q2, r))
    cands = refined_cands
    for quad, r in cands:
        try:
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
    info["edge_refine"] = edge_refine_infos
    try:
        paper2, cinfo = converge_corners(bgr, paper)
    except Exception as exc:
        # Degenerate paper quad: keep the unconverged quad and let the
        # framing layer's own validation (and, failing that, the
        # process_license entry guard) decide the outcome, instead of
        # throwing the photo away here.
        paper2 = paper
        cinfo = {"converge_failed": f"{type(exc).__name__}: {exc}"}
    info = dict(info)
    info["converge_info"] = cinfo
    warp_src = bgr
    warp_quad = paper2
    # Wedge-cut stage bookkeeping (wired 2026-11-14): the quad the
    # delivered warp was sampled from, and the framing block's ink
    # bounds (None when the framing block never ran).
    stage_quad = None
    bounds_r = None

    flat = bool(info.get("flat_scan"))
    if fallback is None and not flat:
        # Local corner refinement (2026-11-12): the converged quad's
        # corners are intersections of two full-length lines descended
        # from the detector's frame chord, and gold-set scoring showed
        # the intersection sliding along curled edges (worst 152 px).
        # Re-fit each corner from the paper boundary in its own
        # neighbourhood before the framing candidates are built on
        # this quad; corners without clean local evidence keep their
        # incoming position. Flat scans (whole image is paper, corners
        # exact) and the frame fallback (no reconciled paper) skip it.
        try:
            paper2, rinfo = refine_corners_local(bgr, paper2, frame_used)
        except Exception as exc:
            rinfo = {"applied": False,
                     "error": f"{type(exc).__name__}: {exc}"}
        info["corner_refine"] = rinfo
        warp_quad = paper2
    warped = None
    if fallback is None and not flat:
        # Three framing candidates, picked by measured evidence on
        # their actual warps -- no candidate wins by construction:
        #   expand   -- the reconciled paper quad as-is;
        #   tighten  -- expand quad + clamp/tighten (lose-a-sliver);
        #   envelope -- straight-line frame envelope, when feasible.
        # Score: QR readable first, then cleanest dark border; a
        # small preference (envelope > tighten > expand) breaks
        # near-ties within 0.02 of border darkness.
        H0, W0 = bgr.shape[:2]
        out_w, out_h = out_size
        dst3 = np.array([[0, 0], [out_w - 1, 0],
                         [out_w - 1, out_h - 1], [0, out_h - 1]],
                        np.float32)

        def _warp3(src_img, q):
            M3 = cv2.getPerspectiveTransform(q.astype(np.float32), dst3)
            return cv2.warpPerspective(src_img, M3, (out_w, out_h))

        def _score3(w):
            db = dark_border_fraction(w)
            worst = max(db.get(k, 0.0)
                        for k in ("top", "bottom", "left", "right"))
            return (1 if _qr_readable(w) else 0), -worst

        fc_q, fc_info = clamp_quad_to_frame((H0, W0), paper2,
                                            frame_used)
        info["frame_clamp"] = fc_info
        te_q, te_info = tighten_edges(bgr, fc_q, frame_used)
        info["tighten"] = te_info
        # The envelope's target margins stay the physical sizes
        # (250/370 output px = 21mm / 31mm on the old 300dpi
        # canvas, scaled with the canvas): the envelope is a
        # candidate GENERATOR working on the photo's real paper,
        # and the electronic standard is enforced at exactly one
        # place -- the wedge stage's repaste (SPEC-FIT audit
        # disposition). A SPEC-FIT trial retargeted the envelope
        # at the standard's pair means (119/166); measured on the
        # 31 gold set it cost +1.4 px of global corner mean (four
        # envelope images +4..+11) for no margin benefit the
        # repaste does not already deliver, so it was reverted
        # the same round -- see SPEC-FIT-RUN.md.
        m_lr = 250.0 * out_w / 3508.0
        m_tb = 370.0 * out_h / 2480.0
        env_q, env_info = envelope_quad(bgr, frame_used, out_size,
                                        m_lr=m_lr, m_tb=m_tb)
        info["envelope"] = env_info

        # Face-content guard (2026-11-13, emblem diagnosis): measure
        # once where the licence face's own ink reaches on each edge
        # (edge_station_bounds on the frame quad). Until now NO
        # framing candidate was ever checked against it -- the score
        # (QR, dark border) is blind to a cut emblem, because cutting
        # INTO the face removes bright content instead of adding a
        # dark band. Two defences below:
        #   guarded -- the tighten quad repaired outward past the
        #     ink line wherever it falls short (guard_quad_to_ink);
        #   veto -- _consider refuses any candidate that still cuts
        #     measured face ink, exactly as it refuses out-of-frame
        #     quads. If every candidate is vetoed, the least-cutting
        #     one is taken as an explicit fallback (recorded), never
        #     silently.
        bounds_r = None
        try:
            _b = edge_station_bounds(bgr, frame_used)
        except Exception:
            _b = None
        if _b is not None:
            _ink, _ext, _oob = _robustify_bounds(*_b[:3])
            bounds_r = (_ink, _ext, _oob, _b[3])

        cands = [("expand", paper2, paper2, 0.0),
                 ("tighten", te_q, te_q, 0.01)]
        if bounds_r is not None:
            g_q, g_counts, g_worst = guard_quad_to_ink(
                frame_used, bounds_r, te_q)
            info["ink_guard"] = {
                "tighten_counts": framing_ink_check(
                    frame_used, bounds_r, te_q)[0],
                "guarded_counts": g_counts,
                "guarded_worst": round(float(g_worst), 1)}
            if g_q is not None and not np.allclose(g_q, te_q):
                cands.append(("guarded", g_q, g_q, 0.015))
        if env_q is not None:
            cands.append(("envelope", env_q, env_q, 0.02))
        best = None
        scores = {}
        vetoed = []

        def _consider(name, q_ret, q_w, pref):
            """Warp and score one candidate. Zero-fabrication guard:
            a quad reaching outside the source image would print
            filled (black) pixels, so the candidate is void on the
            spot, however clean its border would otherwise look.
            Content guard: a candidate that cuts measured face ink
            (framing_ink_check) is likewise void -- face content is
            the one thing framing may never trade away; vetoed
            candidates are remembered for the least-cut fallback."""
            nonlocal best
            if _quad_outside_frame((H0, W0), q_w):
                scores[name] = "oob"
                return None
            if bounds_r is not None:
                counts, worst = framing_ink_check(frame_used,
                                                  bounds_r, q_w)
                if sum(counts) > 0:
                    scores[name] = ["cut", counts,
                                    round(float(worst), 1)]
                    vetoed.append((sum(counts), float(worst), name,
                                   q_ret, q_w, pref))
                    return None
            w = _warp3(bgr, q_w)
            qr, negdark = _score3(w)
            scores[name] = [qr, round(-negdark, 4)]
            key = (qr, negdark + pref)
            if best is None or key > best[0]:
                best = (key, name, q_ret, w)
            return key, negdark

        for name, q_ret, q_w, pref in cands:
            _consider(name, q_ret, q_w, pref)
        if env_q is None and (best is None
                              or -best[0][1] + 0.02 > 0.25):
            # Both legacy framings leave a heavy dark border and the
            # envelope only just missed feasibility: one last-resort
            # tolerance tier (graze 12 / slack 10) may trade a <=12px
            # graze of a content apex against a full-width dark band.
            # It still has to win the same evidence pick, and the
            # zero-fabrication guard applies to it unchanged (graze
            # and slack only ever move the line within the original
            # image's real pixels).
            env_q, env_info = envelope_quad(bgr, frame_used, out_size,
                                            m_lr=m_lr, m_tb=m_tb,
                                            extra_tiers=((12.0, 10.0),))
            info["envelope"] = env_info
            if env_q is not None:
                if best is None:
                    # Every regular candidate was vetoed (content
                    # guard) or void: the desperate envelope competes
                    # through the same guards as everyone else.
                    _consider("envelope", env_q, env_q, 0.02)
                elif _quad_outside_frame((H0, W0), env_q):
                    scores["envelope"] = "oob"
                elif bounds_r is not None and sum(framing_ink_check(
                        frame_used, bounds_r, env_q)[0]) > 0:
                    # The graze tiers may trade a hair of apex for a
                    # clean border among NON-cutting rivals; they may
                    # never outbid a clean winner by cutting content
                    # it preserved.
                    counts, worst = framing_ink_check(frame_used,
                                                      bounds_r, env_q)
                    scores["envelope"] = ["cut", counts,
                                          round(float(worst), 1)]
                else:
                    w = _warp3(bgr, env_q)
                    qr, negdark = _score3(w)
                    scores["envelope"] = [qr, round(-negdark, 4)]
                    key = (qr, negdark + 0.02)
                    # Wide-margin rule: a last-resort graze is only
                    # worth it if it at least halves the dark border.
                    if key > best[0] and -negdark < 0.5 * -best[0][1]:
                        best = (key, "envelope", env_q, w)
                    elif key > best[0]:
                        info["framing_desperate_rejected"] = True
        if best is None and vetoed:
            # Every candidate cut measured face ink somewhere. Take
            # the least-cutting one (fewest violating stations, then
            # shallowest) and SAY so in the info: a recorded partial
            # failure, never a silent one.
            vetoed.sort(key=lambda v: (v[0], v[1]))
            _c, _w, name, q_ret, q_w, _pref = vetoed[0]
            w = _warp3(bgr, q_w)
            qr, negdark = _score3(w)
            best = ((qr, negdark + _pref), name, q_ret, w)
            info["framing_ink_fallback"] = {
                "choice": name, "counts": _c, "worst": round(_w, 1)}
        if best is None:
            # Every candidate sampled outside the source. Paranoia:
            # clamp_quad_to_frame should make this unreachable. Pull
            # the tighten quad's corners into the frame numerically
            # (geometry only) rather than print filled pixels.
            q_clip = te_q.copy()
            q_clip[:, 0] = np.clip(q_clip[:, 0], 0, W0 - 1)
            q_clip[:, 1] = np.clip(q_clip[:, 1], 0, H0 - 1)
            scores["tighten_clipped"] = "fallback"
            best = ((0, 0.0), "tighten_clipped", q_clip,
                    _warp3(bgr, q_clip))
        info["framing"] = {"choice": best[1], "scores": scores}
        paper2 = best[2]
        warped = best[3]
        stage_quad = paper2

    if warped is None:
        if _quad_outside_frame(warp_src.shape[:2], warp_quad):
            # Zero-fabrication guard on the flat/fallback path too:
            # pull a poking quad back inside the source (geometry
            # only) instead of printing filled pixels.
            warp_quad, _ci = clamp_quad_to_frame(
                warp_src.shape[:2], warp_quad, None)
            info["fallback_clamped"] = True
        W, H = out_size
        dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1],
                        [0, H - 1]], np.float32)
        M = cv2.getPerspectiveTransform(warp_quad.astype(np.float32),
                                        dst)
        warped = cv2.warpPerspective(warp_src, M, (W, H))
        stage_quad = warp_quad

    # Crop + repaste + gradient-fill stage (lic_wedgecut, ying's
    # per-edge rule 2026-11-13; prototype gold-set verdict: baseline
    # four metrics zero drift, clutter total -27.9%, zero images
    # worse -- wired 2026-11-14). Runs on the homographic warp,
    # BEFORE the TPS arm below, so its frame guard can use exact
    # warp geometry: the pipeline's frame quad mapped through the
    # same homography, plus the station-measured ink reach the
    # framing guards already computed (rescaled into canvas px at
    # each edge's midpoint). Per edge it crops the output's clutter
    # wedge at the full measured depth (the geometric frame guard
    # is the only stop), repastes onto the standard canvas with
    # the frame centred on the electronic standard's position, and
    # fills the gaps by extending the adjacent paper edge's own
    # colours outward. Flat scans are hard-gated out
    # (pixel-faithful originals); a frame edge mapped outside the
    # canvas refuses inside lic_wedgecut and the warp passes
    # through unchanged. An over-cap fill no longer moves the
    # paste there: it is recorded as an explicit warning inside
    # lic_wedgecut (fill_over_cap_warning), because it means the
    # framing quad is probably off (SPEC-FIT round).
    # paper_quad is deliberately NOT modified: the stage lives
    # purely in output space.
    if flat:
        info["wedge_cut"] = {"applied": False,
                             "reason": "flat_scan_gate"}
    elif stage_quad is None:
        info["wedge_cut"] = {"applied": False,
                             "reason": "no_warp_quad"}
    else:
        try:
            _dst = np.array([[0, 0], [out_size[0] - 1, 0],
                             [out_size[0] - 1, out_size[1] - 1],
                             [0, out_size[1] - 1]], np.float32)
            _M = cv2.getPerspectiveTransform(
                stage_quad.astype(np.float32), _dst)
            _frame_out = cv2.perspectiveTransform(
                frame_used.astype(np.float32).reshape(4, 1, 2),
                _M).reshape(4, 2)
            _reach = None
            if bounds_r is not None:
                _ink_r = bounds_r[0]
                _fgeo = _edge_geometry(frame_used)
                _reach = []
                for _i in range(4):
                    _a, _b, _n, _mid = _fgeo[_i]
                    _r_src = float(np.max(_ink_r[_i])) \
                        if len(_ink_r[_i]) else 0.0
                    _p_out = cv2.perspectiveTransform(
                        np.array([[_mid + _n * _r_src]], np.float32),
                        _M)[0, 0]
                    _m_out = cv2.perspectiveTransform(
                        np.array([[_mid]], np.float32), _M)[0, 0]
                    _reach.append(float(np.hypot(*(_p_out - _m_out))))
            _qr_before = _qr_readable(warped)
            _w2, winfo = crop_repave_fill(warped, _frame_out,
                                          _reach)
            if _qr_before and winfo.get("applied") \
                    and not _qr_readable(_w2):
                # Self-verification, the TPS arm's house pattern:
                # this stage exists to clean margins, and a crop +
                # translation that costs a previously readable QR
                # its decode (a marginal code flips on sampling
                # phase alone -- measured: a 6 px shift flipped
                # Image-14-06-14 both ways) has traded content for
                # cosmetics. Keep the pre-stage warp and SAY so.
                winfo = dict(winfo)
                winfo["applied"] = False
                winfo["accepted"] = False
                winfo["reverted_qr_loss"] = True
            else:
                warped = _w2
        except Exception as exc:
            winfo = {"applied": False,
                     "error": f"{type(exc).__name__}: {exc}"}
        info["wedge_cut"] = winfo

    # TPS dewarp (lic_undistort), photo route only. Flat scans are
    # hard-gated out above all: they are pixel-faithful originals and
    # the arm's ruler reads phantom bow on electronic ornament (8 of 9
    # flat pack members would be remapped by the bare arm). The arm is
    # self-verifying: it returns the warp pixel-identical unless the
    # correction provably reduces the measured bow, so this stage is
    # byte-neutral for every untriggered / rolled-back photo.
    dewarp_skipped_high_bow = False
    if flat:
        info["dewarp"] = {"triggered": False, "reason": "flat_scan_gate"}
    else:
        try:
            warped, dinfo = undistort_license(warped)
        except Exception as exc:
            # The new stage must never cost a finished photo its result.
            dinfo = {"triggered": False, "reason": "error",
                     "error": f"{type(exc).__name__}: {exc}"}
        info["dewarp"] = dinfo
        if not dinfo.get("triggered") \
                and dinfo.get("reason") in ("no_improvement",
                                            "field_too_large") \
                and float(dinfo.get("max_dev_before") or 0.0) \
                > gate_max_dev_px(warped.shape[1]):
            # Bow measured above the gate, but the self-check would
            # not let the arm touch it: flag it so the UI can say so
            # honestly instead of pretending the page came out
            # straightened (review pack n02/n06/n09 are this shape).
            dewarp_skipped_high_bow = True

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
            "warped": warped, "paper_quad": paper2, "fallback": fallback,
            "dewarp_skipped_high_bow": dewarp_skipped_high_bow,
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
      dewarp_skipped_high_bow: True when the TPS ruler measured a bow
        above its gate on the photo route but the self-check refused
        the correction (the bow is still in the output; UI should say
        so or offer manual marking, not pretend otherwise)
      info:   expand/converge/envelope/enhance/dark-border diagnostics;
        info["scrfd_error"] / info["detect_degraded"] are present when
        the SCRFD detector threw or produced no usable candidate and
        the contour routes carried the photo (2026-11-14 hardening);
        info["autorotate"] is present when a sideways source was
        auto-rotated (chosen direction + trigger evidence);
        info["dewarp"] carries the TPS arm's verdict on photo routes
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
