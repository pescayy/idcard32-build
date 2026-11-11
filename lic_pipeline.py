# Licence (business licence, A4/A3 landscape) processing glue for the Qt line.
# Pure function, no UI: app_qt.py calls process_license() and renders the
# result. Chains the modules that are already tested in isolation:
#   quad detect (SCRFD -> contour -> contour on illumination-flattened)
#   -> bigdoc.expand_to_paper_edge (flat scans short-circuit to whole image)
#   -> bigdoc.converge_corners -> perspective warp to output size
#   -> bigdoc.dark_border_fraction report.
# Colour stages (lic_enhance + lic_dehaze) are kept in code but bypassed
# by default since 2026-11-11 (PS logic: automatic processing moves
# pixels, it does not change their values); pass tone=True to run them.
# Flat scans skip them either way (measured warm-paper bleaching).
# A None from expand_to_paper_edge on every candidate quad (curled corner,
# heavy shadow) falls back to the best frame quad, flagged via
# info["expand_fallback"] so the UI can warn and offer manual marking.
import os

import numpy as np
import cv2

from bigdoc import expand_to_paper_edge, converge_corners, dark_border_fraction
from lic_enhance import enhance_license
from lic_dehaze import dehaze_license

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
    if detector is not None:
        try:
            q = detector.detect(bgr)
        except Exception:
            q = None
        if q is not None:
            out.append((order_quad(q), "scrfd"))
    q = contour_detect(bgr)
    if q is not None:
        out.append((order_quad(q), "contour"))
    q = contour_detect(_flatten_illumination(bgr))
    if q is not None:
        out.append((order_quad(q), "contour_flat"))
    return out


def detect_license_quad(bgr, detector=None):
    """Returns (best quad 4x2 TL,TR,BR,BL or None, route name or None)."""
    cands = detect_license_quads(bgr, detector)
    if not cands:
        return None, None
    return cands[0]


def process_license(bgr, detector=None, out_size=OUT_A4, model_path=None,
                    tone=False):
    """Process one licence image.

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
      info:   expand/converge/enhance/dark-border diagnostics
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
                "info": {"reason": "no frame quad detected"}}

    # Try each candidate quad until one reconciles with the A4 paper edge.
    paper, info, route = None, {}, cands[0][1]
    for quad, r in cands:
        paper, info = expand_to_paper_edge(bgr, quad)
        if paper is not None:
            route = r
            break
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

    paper2, cinfo = converge_corners(bgr, paper)
    info = dict(info)
    info["converge_info"] = cinfo

    W, H = out_size
    dst = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]],
                   np.float32)
    M = cv2.getPerspectiveTransform(paper2.astype(np.float32), dst)
    warped = cv2.warpPerspective(bgr, M, (W, H))

    flat = bool(info.get("flat_scan"))
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
            "info": info}
