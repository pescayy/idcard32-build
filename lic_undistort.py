# Non-rigid residual-distortion correction for the licence pipeline.
#
# STATUS 2026-11-11 (later batch): WIRED into lic_pipeline -- called
# after the warp, before toning, photo route only, flat scans
# hard-gated out by the caller. Wiring was approved on the two
# evidence rounds recorded in PORT-STATUS (30-pack batch + 15-photo
# review): triggered corrections always improved the measured bow,
# untriggered / self-check-rejected outputs are pixel-identical, QR
# readability never flipped. The stop-at-the-gate history below is
# why the gate and the self-verify exist in their current shape.
# On Image-14-06-14 the raw-warp ruler reads max 15.7 px (top), and
# every field variant tried (TPS near-exact / smoothed / robust-target,
# separable Coons) improved left/right/bottom but made the top reading
# worse (16.6-19.2): the top statistic is dominated by a glare-zone
# centroid artifact in the guilloche band, not by removable geometry,
# and the built-in verify below correctly reverts such attempts.
# Cross-checks: border-midpoint and centroid estimators disagree by up
# to ~20 px on the same band; the 36.1 px figure recorded earlier was
# measured after LIC_ENH, whose gain field alone inflates the centroid
# reading 15.7 -> 36.6 with zero geometry change. See PORT-STATUS.md
# (2026-11-11 午后六) for the full evidence table.
#
# The perspective warp assumes a perfectly flat sheet. Real phone photos
# keep a smooth, mostly non-radial bow (paper lying not-quite-flat, shot
# up close) that no homography -- and, as measured 2026-11-11, no
# single-parameter radial model either -- can remove. The printed frame
# on the licence is a true rectangle, so its four band centrelines are a
# built-in ruler: fit each edge's straight line, then bend the warped
# image with a thin-plate spline so the centrelines land on their lines
# and the interior follows smoothly.
#
# Contract (lic_pipeline calls this after the warp, before toning, on
# the photo route only; flat scans never reach it):
#   undistort_license(bgr) -> (image, info)
# The image is returned pixel-identical unless the measured bow both
# exceeds the gate and is verifiably reduced by the correction:
#   * ruler incomplete (an edge band not found)  -> pass through
#   * max edge deviation <= gate (physical, see below) -> pass through
#   * fitted field exceeds the displacement cap  -> pass through
#   * re-measured deviation not better than pre  -> pass through
# info always reports the measured per-edge deviations so callers can
# log what the ruler saw.
#
# Threshold units (2026-11-12, gate-tightening batch): the gate, the
# field cap and the control-point pitch were calibrated in pixels of
# the old 3508x2480 (A4 @300dpi) warp canvas. When the output canvas
# was capped at 2339x1654 (A4 @200dpi) those pixel constants silently
# changed physical meaning -- the same physical bow read ~2/3 the
# pixels, so genuinely curled sheets fell below the gate while two
# flat sheets' phantom readings cleared it and got bent. All three
# lengths are now stated in millimetres on the document and converted
# to pixels of whatever canvas the warp actually has (the warped
# width spans the 297mm sheet width), so the arm's behaviour no
# longer depends on the output resolution. _GRID_STEP stays a pixel
# count on purpose: it is the field's evaluation grid, not a
# threshold, and at 8px it remains finer than the control-point
# pitch at either canvas.
import numpy as np
import cv2

DOC_WIDTH_MM = 297.0   # the warped canvas spans the A4 sheet width
GATE_MAX_DEV_MM = 8.0 * 25.4 / 300.0    # 0.677mm; was 8.0px @300dpi
_MAX_DISP_MM = 90.0 * 25.4 / 300.0      # 7.62mm;  was 90.0px @300dpi
_MIN_SPACING_MM = 14.0 * 25.4 / 300.0   # 1.185mm; was 14.0px @300dpi
_GRID_STEP = 8         # TPS evaluated on a coarse grid, then upsampled


def _px_per_mm(width_px):
    """Pixels per millimetre on a warp of the given width."""
    return float(width_px) / DOC_WIDTH_MM


def gate_max_dev_px(width_px):
    """The bow gate in pixels for a warp of the given width."""
    return GATE_MAX_DEV_MM * _px_per_mm(width_px)


# ------------------------------------------------------------------ ruler
# Centreline extraction is the probe ruler of probe_lic_undistort.py,
# copied verbatim so every number in this module is measured with the
# same ruler as the recorded baselines (left 7.8 / right 11.1 /
# top 36.1 / bottom 22.9 px on Image-14-06-14, 3508-wide output).
def _smooth(v, w):
    if w <= 1 or len(v) < 3:
        return v.copy()
    w = int(w) | 1
    k = np.ones(w) / w
    pad = w // 2
    return np.convolve(np.pad(v, (pad, pad), mode="edge"), k,
                       mode="valid")[:len(v)]


def _band_center(prof, lo, hi):
    lo, hi = max(0, lo), min(len(prof), hi)
    if hi - lo < 10:
        return None
    seg = _smooth(prof[lo:hi].astype(np.float64), 15)
    return lo + int(np.argmax(seg))


def _centroids(D, fixed_idx, win_lo, win_hi, axis):
    out = np.full(len(fixed_idx), np.nan)
    coords = np.arange(win_lo, win_hi, dtype=np.float64)
    for i, f in enumerate(fixed_idx):
        line = (D[f, win_lo:win_hi] if axis == 0
                else D[win_lo:win_hi, f]).astype(np.float64)
        t = max(float(line.max()) * 0.30, 10.0)
        w = np.clip(line - t, 0, None)
        s = w.sum()
        if s > 30:
            out[i] = float((w * coords).sum() / s)
    return out


def frame_edge_points(warped):
    """Smoothed frame-band centreline sample points per edge,
    warped coords, as {name: (N,2) array of (x, y)}."""
    H, W = warped.shape[:2]
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY).astype(np.float64)
    D = 255.0 - gray
    gx = np.abs(cv2.Sobel(D, cv2.CV_64F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(D, cv2.CV_64F, 0, 1, ksize=3))
    hw_x, hw_y = int(0.026 * W), int(0.026 * H)
    sm_v, sm_h = max(21, int(0.035 * H)), max(21, int(0.035 * W))
    pts = {}

    def v(name, y0, y1, xlo, xhi):
        ys = np.arange(int(y0 * H), int(y1 * H))
        xc = _band_center(gx[ys, :].mean(0), int(xlo * W), int(xhi * W))
        if xc is None:
            return
        cen = _centroids(D, ys, max(0, xc - hw_x), min(W, xc + hw_x),
                         axis=0)
        ok = np.isfinite(cen)
        p, c = ys[ok].astype(np.float64), _smooth(cen[ok], sm_v)
        pts[name] = np.stack([c, p], 1)[::4]

    def h(name, segs, ylo, yhi):
        allp = []
        for a, b in segs:
            xs = np.arange(int(a * W), int(b * W))
            yc = _band_center(gy[:, xs].mean(1), int(ylo * H),
                              int(yhi * H))
            if yc is None:
                continue
            cen = _centroids(D, xs, max(0, yc - hw_y), min(H, yc + hw_y),
                             axis=1)
            ok = np.isfinite(cen)
            p, c = xs[ok].astype(np.float64), _smooth(cen[ok], sm_h)
            allp.append(np.stack([p, c], 1))
        if allp:
            pts[name] = np.concatenate(allp)[::4]

    v("left", 0.17, 0.83, 0.02, 0.13)
    v("right", 0.17, 0.70, 0.87, 0.985)   # seal sits below 0.70H
    h("top", [(0.10, 0.36), (0.64, 0.90)], 0.04, 0.24)  # emblem gap
    h("bottom", [(0.08, 0.66)], 0.80, 0.99)            # seal gap
    return pts


def _analyse_edges(pts):
    """Per edge: samples, fitted line, max |deviation| over the central
    8-92% of positions -- the probe ruler's deviation definition."""
    out = {}
    for name, p in pts.items():
        if len(p) < 40:
            continue
        if name in ("left", "right"):
            pos, val = p[:, 1], p[:, 0]
        else:
            pos, val = p[:, 0], p[:, 1]
        a, b = np.polyfit(pos, val, 1)
        line = a * pos + b
        res = np.abs(val - line)
        lo, hi = np.percentile(pos, [8, 92])
        m = (pos >= lo) & (pos <= hi)
        out[name] = {"pts": p, "pos": pos, "val": val, "line": line,
                     "coef": (a, b), "max_dev": float(res[m].max())}
    return out


def measure_edges(warped):
    """{edge: max deviation px, ..., 'max_dev': float} -- same numbers
    as probe_lic_undistort.measure_frame_edges."""
    ana = _analyse_edges(frame_edge_points(warped))
    out = {k: v["max_dev"] for k, v in ana.items()}
    if out:
        out["max_dev"] = max(out.values())
    return out


# -------------------------------------------------------------------- TPS
def _tps_solve(ctrl_dst, ctrl_src, lam=1e-4):
    """Thin-plate spline mapping ctrl_dst -> ctrl_src (both (N,2),
    coordinates pre-normalised). Returns (weights (N,2), affine (3,2))."""
    n = len(ctrl_dst)
    d = ctrl_dst[:, None, :] - ctrl_dst[None, :, :]
    r2 = (d * d).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        K = np.where(r2 > 0, r2 * np.log(np.maximum(r2, 1e-12)), 0.0)
    K.flat[::n + 1] += lam
    P = np.concatenate([np.ones((n, 1)), ctrl_dst], 1)
    A = np.zeros((n + 3, n + 3))
    A[:n, :n] = K
    A[:n, n:] = P
    A[n:, :n] = P.T
    Y = np.concatenate([ctrl_src, np.zeros((3, 2))], 0)
    sol = np.linalg.solve(A, Y)
    return sol[:n], sol[n:]


def _tps_eval(model, query):
    """Evaluate a TPS model at query points (M,2) -> (M,2)."""
    w, a, ctrl = model
    d = query[:, None, :] - ctrl[None, :, :]
    r2 = (d * d).sum(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        U = np.where(r2 > 0, r2 * np.log(np.maximum(r2, 1e-12)), 0.0)
    P = np.concatenate([np.ones((len(query), 1)), query], 1)
    return U @ w + P @ a


def _thin_idx(p, spacing):
    """Indices keeping points at least `spacing` px apart along the
    polyline -- one index set, applied to both the measured points and
    their line targets so pairs stay matched."""
    if len(p) <= 2:
        return np.arange(len(p))
    keep = [0]
    acc = 0.0
    for i in range(1, len(p)):
        acc += float(np.hypot(*(p[i] - p[i - 1])))
        if acc >= spacing:
            keep.append(i)
            acc = 0.0
    if keep[-1] != len(p) - 1:
        keep.append(len(p) - 1)
    return np.array(keep)


# -------------------------------------------------------------- main entry
def undistort_license(bgr):
    """Straighten the residual bow of a warped licence image.

    Returns (image, info); image is the input object, pixel-identical,
    whenever the correction does not provably help (see module header).
    """
    info = {"triggered": False}
    H, W = bgr.shape[:2]
    ppm = _px_per_mm(W)
    gate_px = GATE_MAX_DEV_MM * ppm
    info["gate_px"] = gate_px
    ana = _analyse_edges(frame_edge_points(bgr))
    if len(ana) < 4:
        info["reason"] = "ruler_incomplete"
        info["edges_found"] = sorted(ana)
        return bgr, info
    before = {k: v["max_dev"] for k, v in ana.items()}
    info["max_dev_before"] = max(before.values())
    info["edges_before"] = before
    if info["max_dev_before"] <= gate_px:
        info["reason"] = "below_gate"
        return bgr, info

    # Control points: measured centreline samples -> their projections
    # on the fitted straight line (along-edge coordinate kept).
    src_pts, dst_pts = [], []
    for name, e in ana.items():
        p = e["pts"]
        if name in ("left", "right"):
            tgt = np.stack([e["line"], p[:, 1]], 1)
        else:
            tgt = np.stack([p[:, 0], e["line"]], 1)
        idx = _thin_idx(p, _MIN_SPACING_MM * ppm)
        src_pts.append(p[idx])
        dst_pts.append(tgt[idx])
    src = np.concatenate(src_pts)
    dst = np.concatenate(dst_pts)
    info["n_ctrl"] = int(len(src))

    S = float(W)
    model = _tps_solve(dst / S, src / S)
    model = (model[0], model[1], dst / S)

    # Backward map on a coarse grid: for each corrected pixel, where to
    # sample in the bowed image.
    xs = np.arange(0, W, _GRID_STEP, dtype=np.float64)
    ys = np.arange(0, H, _GRID_STEP, dtype=np.float64)
    gx, gy = np.meshgrid(xs, ys)
    query = np.stack([gx.ravel(), gy.ravel()], 1) / S
    out = np.empty_like(query)
    step = 8192
    for i in range(0, len(query), step):
        out[i:i + step] = _tps_eval(model, query[i:i + step])
    mapx = (out[:, 0] * S).reshape(gx.shape).astype(np.float32)
    mapy = (out[:, 1] * S).reshape(gy.shape).astype(np.float32)
    disp = np.hypot(mapx - gx.astype(np.float32),
                    mapy - gy.astype(np.float32))
    info["field_max_disp"] = float(disp.max())
    if info["field_max_disp"] > _MAX_DISP_MM * ppm:
        info["reason"] = "field_too_large"
        return bgr, info
    mapx = cv2.resize(mapx, (W, H), interpolation=cv2.INTER_CUBIC)
    mapy = cv2.resize(mapy, (W, H), interpolation=cv2.INTER_CUBIC)
    fixed = cv2.remap(bgr, mapx, mapy, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_REPLICATE)

    after = measure_edges(fixed)
    info["edges_after"] = after
    info["max_dev_after"] = after.get("max_dev")
    if (info["max_dev_after"] is None
            or info["max_dev_after"] >= info["max_dev_before"]):
        info["reason"] = "no_improvement"
        return bgr, info
    info["triggered"] = True
    return fixed, info
