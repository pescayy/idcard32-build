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
    envelope_quad, is_flat_scan, A4_RATIO
from lic_enhance import enhance_license
from lic_dehaze import dehaze_license
from lic_undistort import undistort_license, GATE_MAX_DEV

OUT_A4 = (3508, 2480)  # copy (副本): 297x210mm @300dpi, landscape
OUT_A3 = (4961, 3508)  # original (正本): 420x297mm @300dpi, landscape

_BASE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL = os.path.join(_BASE, "models", "carddetection_scrfd34gkps.onnx")


def order_quad(pts):
    p = np.asarray(pts, np.float64)
    s = p.sum(1)
    d = np.diff(p, axis=1).ravel()
    return np.array([p[np.argmin(s)], p[np.argmin(d)],
                     p[np.argmax(s)], p[np.argmax(d)]])


def _quad_sane(quad):
    """Non-raising twin of bigdoc._validate_quad: True only for a finite
    4x2 quad with non-zero area. Degenerate detector output (a zero-area
    SCRFD quad reached the pipeline in the 2026-11-11 douyin batch) is
    filtered with this instead of being allowed to raise downstream."""
    try:
        q = np.asarray(quad, dtype=float)
    except Exception:
        return False
    if q.shape != (4, 2) or not np.all(np.isfinite(q)):
        return False

    def _cross2(a, b):
        return float(a[0] * b[1] - a[1] * b[0])
    area2 = abs(_cross2(q[1] - q[0], q[3] - q[0])) + \
        abs(_cross2(q[2] - q[1], q[0] - q[1]))
    return area2 > 0


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


def detect_license_quads(bgr, detector=None):
    """Candidate frame quads (4x2 TL,TR,BR,BL) in priority order, each with
    its route name. SCRFD first, then contour, then contour on the
    illumination-flattened image. May be empty."""
    out = []

    def _add(q, route):
        if q is None:
            return
        oq = order_quad(q)
        if _quad_sane(oq):
            out.append((oq, route))

    if detector is not None:
        try:
            q = detector.detect(bgr)
        except Exception:
            q = None
        _add(q, "scrfd")
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


def _process_with_cands(bgr, cands, detector, out_size, tone):
    """The pipeline proper, given pre-detected candidate quads."""
    # Try each candidate quad until one reconciles with the A4 paper edge.
    paper, info, route = None, {}, cands[0][1]
    frame_used = cands[0][0]
    expand_errors = []
    for quad, r in cands:
        try:
            p_try, i_try = expand_to_paper_edge(bgr, quad)
        except Exception as exc:
            # A degenerate candidate must not kill the photo: log it
            # and let the next candidate try (2026-11-11 douyin n08).
            expand_errors.append(f"{r}: {type(exc).__name__}: {exc}")
            continue
        info = i_try
        if p_try is not None and _quad_sane(p_try):
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

    flat = bool(info.get("flat_scan"))
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
        env_q, env_info = envelope_quad(bgr, frame_used, out_size)
        info["envelope"] = env_info

        cands = [("expand", paper2, paper2, 0.0),
                 ("tighten", te_q, te_q, 0.01)]
        if env_q is not None:
            cands.append(("envelope", env_q, env_q, 0.02))
        best = None
        scores = {}

        def _consider(name, q_ret, q_w, pref):
            """Warp and score one candidate. Zero-fabrication guard:
            a quad reaching outside the source image would print
            filled (black) pixels, so the candidate is void on the
            spot, however clean its border would otherwise look."""
            nonlocal best
            if _quad_outside_frame((H0, W0), q_w):
                scores[name] = "oob"
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
        if best is not None and env_q is None \
                and -best[0][1] + 0.02 > 0.25:
            # Both legacy framings leave a heavy dark border and the
            # envelope only just missed feasibility: one last-resort
            # tolerance tier (graze 12 / slack 10) may trade a <=12px
            # graze of a content apex against a full-width dark band.
            # It still has to win the same evidence pick, and the
            # zero-fabrication guard applies to it unchanged (graze
            # and slack only ever move the line within the original
            # image's real pixels).
            env_q, env_info = envelope_quad(bgr, frame_used, out_size,
                                            extra_tiers=((12.0, 10.0),))
            info["envelope"] = env_info
            if env_q is not None:
                if _quad_outside_frame((H0, W0), env_q):
                    scores["envelope"] = "oob"
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
                > GATE_MAX_DEV:
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
        info["autorotate"] is present when a sideways source was
        auto-rotated (chosen direction + trigger evidence);
        info["dewarp"] carries the TPS arm's verdict on photo routes
    """
    if bgr is None or bgr.ndim != 3:
        raise ValueError("process_license: need a 3-channel BGR image")
    if detector is None:
        from core import CardDetectorAI
        detector = CardDetectorAI(model_path or DEFAULT_MODEL)

    cands = detect_license_quads(bgr, detector)
    if not cands:
        return {"status": "no_quad", "route": None, "flat": False,
                "warped": None, "paper_quad": None, "fallback": None,
                "dewarp_skipped_high_bow": False,
                "info": {"reason": "no frame quad detected"}}

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
                        return base
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

    return _process_with_cands(bgr, cands, detector, out_size, tone)


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
