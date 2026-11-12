# Big-document (A4 license) paper-edge expansion.
# SCRFD locks onto the license's decorative frame; this module walks each
# frame edge outward to find the true paper edge (brightness step), then
# rebuilds the quad from the four shifted edge lines.
import numpy as np
import cv2


def _line(p, q):
    """Line through p,q as (a,b,c) with a*x+b*y+c=0, normalized."""
    a = q[1] - p[1]
    b = p[0] - q[0]
    c = -(a * p[0] + b * p[1])
    n = np.hypot(a, b)
    return np.array([a / n, b / n, c / n])


def _intersect(l1, l2):
    a1, b1, c1 = l1
    a2, b2, c2 = l2
    det = a1 * b2 - a2 * b1
    if abs(det) < 1e-9:
        return None
    x = (b1 * c2 - b2 * c1) / det
    y = (c1 * a2 - c2 * a1) / det
    return np.array([x, y])


def _edge_profile(gray, a, b, centroid, max_out_frac=0.09, step=2.0):
    """Mean brightness along lines parallel to edge a->b, at offsets from
    -6px (inward) to +max_out_frac*len outward. Returns (offsets, means)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    u = b - a
    L = np.hypot(*u)
    u /= L
    n = np.array([-u[1], u[0]])
    mid = (a + b) / 2.0
    if np.dot(n, mid - centroid) < 0:
        n = -n
    ts = np.arange(0.12, 0.88, 0.08)  # sample positions along edge (skip corners)
    offs = np.arange(-6.0, max_out_frac * L + 1e-6, step)
    if len(offs) == 0:
        return offs, np.array([], dtype=float), n
    means = []
    H, W = gray.shape
    for off in offs:
        base = a + n * off
        pts = base[None, :] + ts[:, None] * (b - a)[None, :]
        xs = np.clip(pts[:, 0], 0, W - 1)
        ys = np.clip(pts[:, 1], 0, H - 1)
        vals = cv2.remap(gray, xs.astype(np.float32).reshape(-1, 1),
                         ys.astype(np.float32).reshape(-1, 1),
                         cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE).ravel()
        means.append(float(vals.mean()))
    return offs, np.array(means), n


A4_RATIO = 297.0 / 210.0


def _validate_quad(quad, who):
    q = np.asarray(quad, dtype=float) if quad is not None else None
    if q is None or q.shape != (4, 2) or not np.all(np.isfinite(q)):
        raise ValueError(f"{who}: quad must be a finite 4x2 array, got {quad!r}")
    def _cross2(a, b):
        return float(a[0] * b[1] - a[1] * b[0])
    area2 = abs(_cross2(q[1] - q[0], q[3] - q[0])) + \
        abs(_cross2(q[2] - q[1], q[0] - q[1]))
    if area2 <= 0:
        raise ValueError(f"{who}: degenerate quad (zero area): {q.tolist()}")
    return q


def _rotation_deg(quad):
    """Angle of the quad's long top edge vs horizontal, in degrees."""
    q = np.asarray(quad, float)
    top = q[1] - q[0]
    return float(np.degrees(np.arctan2(top[1], top[0])))


def border_white_frac(bgr, band_frac=0.04, white=235):
    """Fraction of the outer border band that is (near-)white paper."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    b = max(2, int(band_frac * min(W, H)))
    border = np.concatenate([gray[:b, :].ravel(), gray[H - b:, :].ravel(),
                             gray[:, :b].ravel(), gray[:, W - b:].ravel()])
    return float((border >= white).mean())


def is_flat_scan(bgr, quad, rot_max_deg=2.0, white_min=0.90):
    """True for an electronic file / flat scan: the image itself is the paper
    (near-zero rotation, border ~pure white). For these, expanding outward
    from the frame is meaningless -- the frame is NOT A4-proportioned
    (measured ~1.69 on the 2026-11-10 electronic batch: top/bottom frame
    margins total ~26% of height vs ~11% left/right), so the A4 aspect check
    can never reconcile and expand_to_paper_edge returns None. Callers must
    use the whole image as the paper instead (see expand_to_paper_edge)."""
    if quad is None:
        return False
    try:
        rot = abs(_rotation_deg(quad))
    except Exception:
        return False
    return rot <= rot_max_deg and border_white_frac(bgr) >= white_min


def _edge_candidates(gray, a, b, centroid, perp_len, top_k=3):
    """Candidate outward offsets for one edge: local maxima of the brightness
    profile gradient at offset >= 5px (skips the frame's own border step).
    Offsets are capped by the physical layout: frame->paper margin on an A4
    license is ~3.8% of the short side, so anything beyond 6.5% of the
    perpendicular size is vetoed (bright-desk overshoot guard).
    Always includes the layout default (3.5% of perp) as fallback."""
    cap = 0.11 * perp_len
    offs, means, n = _edge_profile(gray, a, b, centroid,
                                   max_out_frac=cap / max(1.0, float(np.hypot(*(np.asarray(b, float) - np.asarray(a, float))))))
    if len(means) == 0:
        # No profile to read (degenerate edge): fall back to layout default.
        return [(0.035 * perp_len, 0.0)], n
    grad = np.abs(np.gradient(means, offs))
    cand = []
    for i in range(1, len(offs) - 1):
        if offs[i] < 5.0 or offs[i] > cap:
            continue
        if grad[i] >= grad[i - 1] and grad[i] >= grad[i + 1] and grad[i] > 0:
            cand.append((float(offs[i]), float(grad[i])))
    cand.sort(key=lambda t: -t[1])
    cand = cand[:top_k]
    cand.append((0.035 * perp_len, 0.0))  # layout default, zero strength
    return cand, n


def _quad_from_offsets(edges_pts, normals, offsets):
    lines = []
    for (a, b), n, off in zip(edges_pts, normals, offsets):
        p_new = np.asarray(a, float) + n * off
        q_new = np.asarray(b, float) + n * off
        lines.append(_line(p_new, q_new))
    corners = []
    for i in range(4):
        pt = _intersect(lines[(i - 1) % 4], lines[i])
        if pt is None:
            return None
        corners.append(pt)
    return np.array(corners)


def _aspect(quad):
    w = np.hypot(*(quad[1] - quad[0]))
    h = np.hypot(*(quad[3] - quad[0]))
    return max(w, h) / max(1.0, min(w, h))


def expand_to_paper_edge(bgr, quad, target_ratio=A4_RATIO):
    """quad: 4x2 ordered (TL,TR,BR,BL) frame quad from the detector.
    Picks per-edge expansion candidates whose combination best matches the
    A4 aspect ratio (tie-break: stronger total gradient evidence).
    Returns (paper_quad or None, info).

    Flat scans / electronic files short-circuit: the whole image IS the paper,
    so the paper quad is the image corners and no expansion is attempted.
    A None return is a legitimate "could not reconcile with A4" -- callers
    must NOT feed it into converge_corners/warp (that used to crash with a
    cryptic numpy AxisError); fall back to the frame quad or manual marking.
    """
    q = _validate_quad(quad, "expand_to_paper_edge")
    if is_flat_scan(bgr, q):
        H, W = bgr.shape[:2]
        paper = np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], float)
        return paper, {"flat_scan": True, "skipped_expansion": True,
                       "border_white_frac": border_white_frac(bgr),
                       "expansions_px": [0.0, 0.0, 0.0, 0.0],
                       "aspect": float(W / H), "aspect_dev": 0.0, "strength": 0.0}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    q = [np.asarray(p, float) for p in quad]
    centroid = np.mean(q, axis=0)
    edges = [(q[0], q[1]), (q[1], q[2]), (q[2], q[3]), (q[3], q[0])]
    # perpendicular size per edge = distance between the two parallel edges
    perp = []
    for i, (a, b) in enumerate(edges):
        mid = (np.asarray(a, float) + np.asarray(b, float)) / 2.0
        oa, ob = edges[(i + 2) % 4]
        omid = (np.asarray(oa, float) + np.asarray(ob, float)) / 2.0
        perp.append(float(np.hypot(*(mid - omid))))
    cand_lists, normals = [], []
    for (a, b), pl in zip(edges, perp):
        cand, n = _edge_candidates(gray, a, b, centroid, pl)
        cand_lists.append(cand)
        normals.append(n)
    best = None
    import itertools
    for combo in itertools.product(*[range(len(c)) for c in cand_lists]):
        offs = [cand_lists[i][combo[i]][0] for i in range(4)]
        strength = sum(cand_lists[i][combo[i]][1] for i in range(4))
        paper = _quad_from_offsets(edges, normals, offs)
        if paper is None or not np.all(np.isfinite(paper)):
            # A degenerate edge (coincident points) makes _line divide
            # by a zero norm and the intersections come out NaN. Such a
            # combo is not a candidate at all -- previously it could be
            # returned as the "best" paper (NaN keys never lose a
            # comparison) and crashed converge_corners downstream
            # (2026-11-11 batch, douyin n01/n12). Skipping it cannot
            # change any finite outcome: a NaN key never displaces a
            # finite best, and a finite key never displaces a NaN best
            # either -- so whenever the old code returned a finite
            # paper, no NaN combo had won.
            continue
        dev = abs(_aspect(paper) - target_ratio)
        # Selection priors, in order:
        # 1) quad must reconcile with the A4 aspect ratio (~0.7% granularity);
        # 2) the printed frame is centered on the sheet, so opposite margins
        #    are near-equal -- penalize asymmetry (kills runaway edges that
        #    walked onto an underlying white sheet/desk panel);
        # 3) then prefer the tightest total expansion, then strongest evidence.
        # A slightly loose fit beats a refusal: worst case it carries a thin
        # background sliver, while a refusal costs a full manual fallback.
        asym = (abs(offs[0] - offs[2]) / perp[0] + abs(offs[1] - offs[3]) / perp[1])
        total_off = float(sum(offs))
        key = (round(dev, 2), round(asym, 2), total_off, -strength)
        if best is None or key < best[0]:
            best = (key, paper, offs, strength, dev)
    if best is None:
        return None, {}
    _, paper, offs, strength, dev = best
    info = {"expansions_px": [float(o) for o in offs],
            "aspect": float(_aspect(paper)), "aspect_dev": float(dev),
            "strength": float(strength)}
    if dev > 0.06:  # could not reconcile with A4 at all: distrust
        return None, info
    return paper, info


# ---------- corner convergence + dark-edge guard ----------

def _paper_fraction(gray, center, r, paper_level):
    x, y = int(center[0]), int(center[1])
    H, W = gray.shape
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = gray[y0:y1, x0:x1]
    return float((patch >= paper_level).mean())


def converge_corners(bgr, quad, max_shift_frac=0.02):
    """Pull any corner that landed off-paper (e.g. pushed into a dark cloth
    by a long opposite edge line) back along the diagonal toward the centroid
    until its neighborhood is mostly paper. Paper level is estimated from the
    quad interior (bright zone), so shadowed paper still counts as paper.
    Raises ValueError on a None/degenerate quad -- never pass the None that
    expand_to_paper_edge returns on A4 distrust (that used to surface as a
    cryptic numpy 'axis 0 is out of bounds' error)."""
    q = _validate_quad(quad, "converge_corners")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    q = np.asarray(q, float).copy()
    centroid = q.mean(axis=0)
    # paper brightness reference: sample grid well inside the quad
    M = cv2.getPerspectiveTransform(q.astype(np.float32),
                                    np.array([[0, 0], [99, 0], [99, 99], [0, 99]], np.float32))
    inner = cv2.warpPerspective(gray, M, (100, 100))
    paper_level = float(np.percentile(inner[15:85, 15:85], 55)) * 0.72
    w = float(np.hypot(*(q[1] - q[0])))
    r = max(6, int(0.008 * w))
    moved = []
    for i in range(4):
        c = q[i].copy()
        step_dir = centroid - c
        step_dir /= max(1e-6, np.hypot(*step_dir))
        shift = 0.0
        max_shift = max_shift_frac * w
        while _paper_fraction(gray, c, r, paper_level) < 0.5 and shift < max_shift:
            c = c + step_dir * (0.0025 * w)
            shift += 0.0025 * w
        moved.append(float(shift))
        q[i] = c
    return q, {"corner_shifts_px": moved, "paper_level": paper_level}


# ---------- local corner refinement (2026-11-12) ----------
# The paper quad's corners are intersections of two full-length edge
# lines whose angles descend from the detector's frame chord. Gold-set
# scoring (license-goldset-v2, 22 photo samples) showed the corner
# error concentrates exactly there: the detector chord is off by
# 30-90 px at the top corners, expand_to_paper_edge shifts the two
# lines outward by scalar offsets, and their intersection slides along
# the edge -- worst case 152 px (Image-14-53-06 TL), 128 px on the
# scalloped corner of shadow_purple that the three-way comparison
# flagged as a corner-refinement (edge-hugging fit) failure, not a
# paper-finding failure. Bottom corners, where the chord happens to
# sit well, stay at 10-45 px through the same stages.
# refine_corners_local re-fits each corner from the paper boundary in
# the corner's own neighbourhood. The boundary points come from
# edge_station_bounds (the envelope's station march: per-station paper
# extent along each frame edge, with shadow-plateau adaptation and
# sustained-paper tests -- the most battle-tested boundary reader in
# this module). For each corner, the boundary points on the two
# incident edges' corner-adjacent station stretches (10%-38% of the
# edge from the corner, staying off the corner turn itself) are
# robust-line-fitted (PCA + one MAD rejection pass) and the corner is
# the two local lines' intersection -- the same construction the gold
# labeler used ("intersection of the two band outer edges"). A corner
# moves only when both incident fits are clean (enough stations, low
# residual, sane angle) AND the intersection is stable when each
# segment is re-fitted from its near/far halves alone; otherwise it
# keeps its incoming position --
# an unreadable corner is left alone, never guessed. The whole quad
# is returned untouched if the refined quad fails sanity / convexity /
# area checks. (First version note: a hand-rolled per-station
# brightness-step finder was tried and rejected on the evidence --
# too weak on low-contrast edges to fire where needed, and where the
# background offered a cleaner straight feature than the paper edge
# it fired wrongly, Image-14-55-27 TR 121 -> 174 px.)

def _fit_line_pca(pts):
    """Total-least-squares line through pts; returns (point, unit dir,
    rms residual) or None."""
    P = np.asarray(pts, float)
    if len(P) < 3:
        return None
    c = P.mean(axis=0)
    X = P - c
    try:
        _u, _s, vt = np.linalg.svd(X, full_matrices=False)
    except np.linalg.LinAlgError:
        return None
    d = vt[0]
    nrm = np.array([-d[1], d[0]])
    res = np.abs(X @ nrm)
    return c, d, float(np.sqrt((res ** 2).mean())), res


def _fit_line_robust(pts):
    """PCA line fit with one MAD-based outlier rejection pass."""
    got = _fit_line_pca(pts)
    if got is None:
        return None
    c, d, _rms, res = got
    med = float(np.median(res))
    mad = float(np.median(np.abs(res - med))) * 1.4826
    thr = max(3.0, med + 3.0 * max(mad, 0.5))
    keep = res <= thr
    if int(keep.sum()) >= 4 and int(keep.sum()) < len(pts):
        got2 = _fit_line_pca(np.asarray(pts, float)[keep])
        if got2 is not None:
            return got2[0], got2[1], got2[2], int(keep.sum())
    return c, d, got[2], len(pts)


def refine_corners_local(bgr, quad, frame_quad=None):
    """Re-fit each corner of a paper quad from the paper boundary in
    the corner's neighbourhood (see the section header). Returns
    (quad, info); the input quad is returned unchanged when nothing
    qualifies -- this stage only ever moves a corner onto measured
    local evidence."""
    q = _validate_quad(quad, "refine_corners_local")
    if frame_quad is None:
        return q, {"applied": False, "reason": "no_frame_quad"}
    bounds = edge_station_bounds(bgr, np.asarray(frame_quad, float))
    if bounds is None:
        return q, {"applied": False, "reason": "no_paper_reference"}
    ink_all, ext_all, oob_all, _lvl = bounds
    fgeo = _edge_geometry(np.asarray(frame_quad, float))
    geo = _edge_geometry(q)
    samples = len(ext_all[0])
    ts_all = np.linspace(0.04, 0.96, samples)
    out = q.copy()
    detail = []
    cos_min = float(np.cos(np.radians(12.0)))

    def _seg_line(pts, u):
        if len(pts) < 4:
            return None
        fit = _fit_line_robust(pts)
        if fit is None:
            return None
        c0, d0, rms, _nu = fit
        if rms > 6.0 or abs(float(np.dot(d0, u))) < cos_min:
            return None
        return _line(c0, c0 + d0), rms

    # Segment pre-pass: for each edge, fit the boundary points in the
    # station stretch nearest each end (10%-38% of the edge from the
    # corner, staying off the corner turn itself).
    seg_info = {}
    for j in range(4):
        a, b, n, _mid = fgeo[j]
        seg = b - a
        qa, qb, _qn, _qm = geo[j]
        u = qb - qa
        u = u / max(1e-9, float(np.hypot(*u)))
        for at_b_end in (False, True):
            if at_b_end:
                sel = (ts_all >= 0.62) & (ts_all <= 0.90)
            else:
                sel = (ts_all >= 0.10) & (ts_all <= 0.38)
            pts = []
            for k in np.nonzero(sel)[0]:
                if oob_all[j][k]:
                    continue        # extent set by the image border
                e = float(ext_all[j][k])
                if e <= 0.0:
                    continue
                pts.append(a + seg * float(ts_all[k]) + n * e)
            entry = {"pts": pts, "line": None, "u": u}
            if len(pts) >= 5:
                got = _seg_line(pts, u)
                if got is not None:
                    entry["line"], entry["rms"] = got
            seg_info[(j, at_b_end)] = entry
    # Edge consistency: a paper edge is ONE straight line (the cutting
    # rule is four straight sides). If the fits from the edge's two
    # ends disagree -- angle over 3 degrees, or the two lines sitting
    # more than 12 px apart at the edge's midpoint -- then at least
    # one end measured something else (a twisted tail segment near a
    # lifted corner reads as a clean but different line), and neither
    # end may move its corner.
    for j in range(4):
        e0 = seg_info[(j, False)]
        e1 = seg_info[(j, True)]
        if e0["line"] is None or e1["line"] is None:
            continue
        d0 = np.array([-e0["line"][1], e0["line"][0]])
        d1 = np.array([-e1["line"][1], e1["line"][0]])
        cosang = abs(float(np.dot(d0, d1)))
        # Separation is judged at the edge's midpoint: two fits of one
        # straight edge under perspective/noise differ by a degree or
        # two and separate slowly (a far-centroid comparison would
        # veto them all), while a segment that measured a different,
        # parallel line (a twisted tail) is offset everywhere,
        # midpoint included.
        mid = fgeo[j][3]
        L0, L1 = e0["line"], e1["line"]
        p0 = mid - (L0[0] * mid[0] + L0[1] * mid[1] + L0[2]) \
            * np.array([L0[0], L0[1]])
        p1 = mid - (L1[0] * mid[0] + L1[1] * mid[1] + L1[2]) \
            * np.array([L1[0], L1[1]])
        sep = max(abs(float(L1[0] * p0[0] + L1[1] * p0[1] + L1[2])),
                  abs(float(L0[0] * p1[0] + L0[1] * p1[1] + L0[2])))
        if cosang < float(np.cos(np.radians(3.0))) or sep > 12.0:
            e0["line"] = e1["line"] = None
            e0["veto"] = e1["veto"] = "edge_inconsistent"
            detail.append({"edge": j, "reason": "edge_inconsistent",
                           "cosang": round(cosang, 4),
                           "sep": round(sep, 1)})
    for i in range(4):
        C = q[i]
        prev_pt = q[(i - 1) % 4]
        next_pt = q[(i + 1) % 4]
        variants = []
        ok = True
        # Each segment is also fitted from its near/far halves
        # separately, because the corner is only trustworthy when the
        # intersection is STABLE under that window perturbation: a
        # boundary that flares, curls or changes surface inside the
        # window shifts the half-fits apart, while a straight
        # boundary stretch gives the same line from every sub-window.
        for j, at_b_end in (((i - 1) % 4, True), (i, False)):
            entry = seg_info[(j, at_b_end)]
            pts = entry["pts"]
            if entry["line"] is None:
                ok = False
                detail.append({"corner": i, "edge": j,
                               "reason": entry.get("veto",
                                                   "few_stations"
                                                   if len(pts) < 5
                                                   else "fit_rejected"),
                               "n": len(pts)})
                break
            seg_vars = [entry["line"]]
            detail.append({"corner": i, "edge": j,
                           "rms": round(entry["rms"], 2),
                           "n": len(pts)})
            half = len(pts) // 2
            for sub in (pts[:max(half, len(pts) - half)],
                        pts[min(half, len(pts) - half):]):
                got_h = _seg_line(sub, entry["u"])
                if got_h is not None:
                    seg_vars.append(got_h[0])
            variants.append(seg_vars)
        if not ok or len(variants) != 2:
            continue
        pt = _intersect(variants[0][0], variants[1][0])
        if pt is None:
            continue
        # stability: every variant pairing's intersection must stay
        # near the full-fit corner; a corner whose position depends
        # on which sub-window measured it is not evidence, it is a
        # guess -- keep the incoming corner instead.
        cands_pt = []
        for la in variants[0]:
            for lb in variants[1]:
                p_ab = _intersect(la, lb)
                if p_ab is not None:
                    cands_pt.append(p_ab)
        if len(cands_pt) >= 4:
            spread = max(float(np.hypot(*(p_ab - pt)))
                         for p_ab in cands_pt)
            if spread > 20.0:
                detail.append({"corner": i, "reason": "unstable",
                               "spread": round(spread, 1)})
                continue
        move = float(np.hypot(*(pt - C)))
        cap = 0.18 * min(float(np.hypot(*(C - prev_pt))),
                         float(np.hypot(*(next_pt - C))))
        if move > cap:
            detail.append({"corner": i, "reason": "move_capped",
                           "move": round(move, 1), "cap": round(cap, 1)})
            continue
        out[i] = pt
        detail.append({"corner": i, "moved_px": round(move, 1)})
    if np.allclose(out, q):
        return q, {"applied": False, "detail": detail}
    # whole-quad sanity: finite, sane area vs input, convex, same winding
    try:
        _validate_quad(out, "refine_corners_local(out)")
    except ValueError:
        return q, {"applied": False, "detail": detail,
                   "reason": "output_insane"}

    def _area(p):
        return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                               np.dot(p[:, 1], np.roll(p[:, 0], -1))))
    if not (0.70 * _area(q) <= _area(out) <= 1.40 * _area(q)):
        return q, {"applied": False, "detail": detail,
                   "reason": "area_out_of_band"}
    signs = []
    for i in range(4):
        e1 = out[(i + 1) % 4] - out[i]
        e2 = out[(i + 2) % 4] - out[(i + 1) % 4]
        signs.append(float(e1[0] * e2[1] - e1[1] * e2[0]))
    if not (all(s > 0 for s in signs) or all(s < 0 for s in signs)):
        return q, {"applied": False, "detail": detail,
                   "reason": "non_convex"}
    return out, {"applied": True, "detail": detail}


# ---------- out-of-frame clamp + evidence tightening (2026-11-11) ----------
# Qing's rule: rather lose a sliver of paper than show anything outside it.
# warpPerspective fills out-of-source samples with black, so a paper quad
# that pokes past the image border prints a black band; and a straight quad
# laid across a curled/wavy sheet edge swallows a strip of background.
# Both fixes only move edge lines inward -- pixel values are never touched
# -- and an edge may never cross the printed frame (minus a reserve).

def _edge_geometry(quad):
    """Per-edge (a, b, outward unit normal, midpoint), TL,TR,BR,BL order."""
    q = [np.asarray(p, float) for p in quad]
    centroid = np.mean(q, axis=0)
    out = []
    for i in range(4):
        a, b = q[i], q[(i + 1) % 4]
        u = b - a
        u = u / max(1e-9, float(np.hypot(*u)))
        n = np.array([-u[1], u[0]])
        mid = (a + b) / 2.0
        if np.dot(n, mid - centroid) < 0:
            n = -n
        out.append((a, b, n, mid))
    return out


def _rebuild_from_edge_shifts(quad, shifts):
    """Move each edge line along its outward normal by shifts[i] (negative
    = inward), rebuild corners by line intersection. None if degenerate."""
    q = np.asarray(quad, float)
    lines = []
    for (a, b, n, _mid), s in zip(_edge_geometry(q), shifts):
        lines.append(_line(a + n * s, b + n * s))
    corners = []
    for i in range(4):
        pt = _intersect(lines[(i - 1) % 4], lines[i])
        if pt is None:
            return None
        corners.append(pt)
    out = np.array(corners)
    if not np.all(np.isfinite(out)):
        return None

    def _area(p):
        return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1)) -
                               np.dot(p[:, 1], np.roll(p[:, 0], -1))))
    if _area(out) < 0.5 * _area(q):
        return None
    return out


def _perp_sizes(quad):
    geo = _edge_geometry(quad)
    return [float(np.hypot(*(geo[i][3] - geo[(i + 2) % 4][3])))
            for i in range(4)]


def _frame_reserve(quad, frame_quad):
    """Per-edge travel budget (px): distance from the paper edge line to
    the frame edge line, at the paper edge midpoint. +inf without a frame."""
    if frame_quad is None:
        return [float("inf")] * 4
    fgeo = _edge_geometry(np.asarray(frame_quad, float))
    res = []
    for i, (_a, _b, _n, mid) in enumerate(_edge_geometry(np.asarray(quad, float))):
        fl = _line(fgeo[i][0], fgeo[i][1])
        res.append(abs(float(fl[0] * mid[0] + fl[1] * mid[1] + fl[2])))
    return res


def _max_penetration(shape_hw, quad):
    """Worst out-of-image penetration (px, axis-wise) over the quad edges."""
    H, W = int(shape_hw[0]), int(shape_hw[1])
    worst = 0.0
    for a, b, _n, _m in _edge_geometry(quad):
        ts = np.linspace(0.0, 1.0, 33)
        pts = a[None, :] + ts[:, None] * (b - a)[None, :]
        pen = np.maximum.reduce([
            -pts[:, 0], pts[:, 0] - (W - 1),
            -pts[:, 1], pts[:, 1] - (H - 1), np.zeros(len(pts))])
        worst = max(worst, float(pen.max()))
    return worst


def clamp_quad_to_frame(shape_hw, quad, frame_quad=None, safety_px=1.5,
                        reserve_px=2.0):
    """Pull any paper edge that pokes outside the image back inside.

    Each offending edge line shifts inward by its worst penetration +
    safety_px, capped so it never crosses the frame edge (minus
    reserve_px) and never beyond 6% of the perpendicular size. Returns
    (quad, info); the input values are returned untouched when nothing
    pokes out, so in-frame quads keep bit-identical downstream output.
    """
    q = _validate_quad(quad, "clamp_quad_to_frame")
    H, W = int(shape_hw[0]), int(shape_hw[1])
    perp = _perp_sizes(q)
    reserve = _frame_reserve(q, frame_quad)
    detail = [{"oob_px": 0.0, "shift_px": 0.0, "capped": False}
              for _ in range(4)]
    cur = q
    applied = False
    for _pass in range(2):
        geo = _edge_geometry(cur)
        shifts = [0.0] * 4
        for i, (a, b, n, _mid) in enumerate(geo):
            # An edge only answers for the image side(s) it actually
            # faces (per its outward normal); a corner poking out along
            # the OTHER axis is the neighbouring edge's shift, which
            # slides this corner along this edge's own line.
            ts = np.linspace(0.0, 1.0, 33)
            pts = a[None, :] + ts[:, None] * (b - a)[None, :]
            viols = [np.zeros(len(pts))]
            if abs(n[0]) >= 0.5:
                viols += [-pts[:, 0], pts[:, 0] - (W - 1)]
            if abs(n[1]) >= 0.5:
                viols += [-pts[:, 1], pts[:, 1] - (H - 1)]
            worst = float(np.maximum.reduce(viols).max())
            if worst <= 0.0:
                continue
            want = worst + float(safety_px)
            # Containment is a measured physical need (those source
            # pixels do not exist), so it may override the frame reserve
            # by a few px -- losing a sliver of frame ornament beats
            # printing a black band. The 6%-of-perp sanity cap stays.
            res_cap = reserve[i] - reserve_px - detail[i]["shift_px"]
            cap = max(0.0, min(0.06 * perp[i], max(res_cap, want)))
            t = min(want, cap)
            shifts[i] = -t
            d = detail[i]
            d["oob_px"] = round(max(d["oob_px"], worst), 2)
            d["shift_px"] = round(d["shift_px"] + t, 2)
            d["capped"] = bool(d["capped"] or t < want - 1e-9)
            if want > res_cap:
                d["reserve_overridden"] = True
        if all(s == 0.0 for s in shifts):
            break
        nxt = _rebuild_from_edge_shifts(cur, shifts)
        if nxt is None:
            return q, {"applied": applied, "edges": detail,
                       "rebuild_failed": True}
        cur = nxt
        applied = True
        if _max_penetration(shape_hw, cur) <= 0.0:
            break
    if not applied:
        return q, {"applied": False, "edges": detail}
    return cur, {"applied": True, "edges": detail,
                 "residual_oob_px": round(_max_penetration(shape_hw, cur), 2)}


def tighten_edges(bgr, quad, frame_quad=None, max_frac=0.012,
                  reserve_px=2.0, min_hit_frac=0.10):
    """Shrink each paper edge inward past off-paper intrusion (background
    a straight quad swallows along a curled/wavy sheet edge).

    Per edge, stations along the edge test successive inward depths: a
    station's intrusion depth is where the band just inside first reads
    as paper for two consecutive steps. The paper test is stricter than
    converge_corners' (0.82 of the interior level, not 0.72): mid-tone
    background (cardboard, desk wood) must count as intrusion, and under
    the lose-a-sliver rule a false nibble of shadowed margin costs less
    than a swallowed background strip. The edge shifts by the
    90th-percentile intrusion when at least min_hit_frac of its stations
    intrude >= 2px; the shift is capped by max_frac of the perpendicular
    size and by the frame reserve. The shift is the 90th percentile over
    the intruding stations only (a percentile over all stations dilutes
    a local curl zone back into the debris). Clean edges do not move.
    Returns (quad, info); input untouched when nothing qualifies.
    """
    q = _validate_quad(quad, "tighten_edges")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    M = cv2.getPerspectiveTransform(
        q.astype(np.float32),
        np.array([[0, 0], [99, 0], [99, 99], [0, 99]], np.float32))
    inner = cv2.warpPerspective(gray, M, (100, 100))
    paper_level = float(np.percentile(inner[15:85, 15:85], 55)) * 0.82
    geo = _edge_geometry(q)
    perp = _perp_sizes(q)
    reserve = _frame_reserve(q, frame_quad)
    r = max(3, int(0.004 * float(np.hypot(*(q[1] - q[0])))))
    shifts = [0.0] * 4
    detail = []
    for i, (a, b, n, _mid) in enumerate(geo):
        bound = min(max_frac * perp[i], max(0.0, reserve[i] - reserve_px))
        if bound < 1.0:
            detail.append({"bound_px": round(float(bound), 2),
                           "intrusion_p90": 0.0, "hit_frac": 0.0,
                           "shift_px": 0.0})
            continue
        nin = -n
        ts = np.linspace(0.05, 0.95, 41)
        pts0 = a[None, :] + ts[:, None] * (b - a)[None, :]
        steps = int(np.ceil(bound))
        intr = []
        for p in pts0:
            fracs = []
            for d in range(0, steps + 1):
                c = p + nin * (float(d) + max(2, r // 2))
                fracs.append(_paper_fraction(gray, c, r, paper_level))
            d_hit = float(bound)
            for d in range(len(fracs)):
                if fracs[d] >= 0.6 and \
                        (d + 1 >= len(fracs) or fracs[d + 1] >= 0.6):
                    d_hit = float(d)
                    break
            intr.append(d_hit)
        intr = np.array(intr)
        hit_frac = float((intr >= 2.0).mean())
        hits = intr[intr >= 2.0]
        # Shift by the deep end of the HIT stations only: a curled zone
        # typically covers a minority of stations, so a percentile over
        # all stations dilutes it back into the debris (measured: sag of
        # 9px read as p90 5px over the full edge, 9px over the hits).
        p90 = float(np.percentile(hits, 90)) if len(hits) else 0.0
        t = 0.0
        if hit_frac >= min_hit_frac and p90 >= 2.0:
            t = min(p90, bound)
        shifts[i] = -t
        detail.append({"bound_px": round(float(bound), 2),
                       "intrusion_p90": round(p90, 2),
                       "hit_frac": round(hit_frac, 3),
                       "shift_px": round(float(t), 2)})
    info = {"edges": detail, "paper_level": round(paper_level, 1)}
    if all(s == 0.0 for s in shifts):
        return q, dict(info, applied=False)
    out = _rebuild_from_edge_shifts(q, shifts)
    if out is None:
        return q, dict(info, applied=False, rebuild_failed=True)
    return out, dict(info, applied=True)


def _dark_fraction(gray, center, r, dark_level):
    x, y = int(center[0]), int(center[1])
    H, W = gray.shape
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    patch = gray[y0:y1, x0:x1]
    return float((patch < dark_level).mean())


def _paper_fraction_sat(gray, sat, center, r, paper_level, sat_max=110):
    """Paper test with a saturation guard: bright AND not vivid.

    The gilt emblem that rides on a licence's top frame is bright in
    gray (gold reads up to ~200) but highly saturated; paper is not.
    A gray-only test mistakes the emblem for paper and the march
    stops inside it.
    """
    x, y = int(center[0]), int(center[1])
    H, W = gray.shape
    x0, x1 = max(0, x - r), min(W, x + r + 1)
    y0, y1 = max(0, y - r), min(H, y + r + 1)
    if x1 <= x0 or y1 <= y0:
        return 0.0
    ok = (gray[y0:y1, x0:x1] >= paper_level) & (sat[y0:y1, x0:x1] <= sat_max)
    return float(ok.mean())


def edge_station_bounds(bgr, frame_quad, samples=41):
    """Per-station outward bounds along each frame edge (source px).

    Marching outward from the frame edge line (into the paper margin),
    each station yields:

    * ``ink_outer`` -- the licence face's own outermost content. The
      detected quad is the chord between the frame's corners; the
      printed frame bows on a curled sheet, and face content can ride
      on the frame itself (the gilt emblem straddles the top frame
      band, tens of px past the chord). A dark OR saturated run that
      paper resumes after, within a long look-ahead, is that content:
      the framing line must move OUT past it.
    * ``paper_extent`` -- where sustained paper ends (desk, clutter,
      a curl's intrusion zone, or the image border): the framing line
      must stay INSIDE it. Paper is first tested strictly (0.82 of
      the interior level, low saturation); a uniform mid-tone plateau
      contiguous with the frame may establish a lower local reference
      (paper lying in the sheet's own shadow reads ~0.6 of the lit
      interior and no absolute level will ever call it paper), after
      which the march follows it adaptively (0.78 of the local
      reference, never below 0.55 of the global one).
    * ``oob`` -- the extent was set by the image border itself (the
      sheet was photographed flush with the frame edge).

    Returns (ink, extent, oob, paper_level): three lists of 4 arrays
    (edges top, right, bottom, left) plus the strict level; None if
    the interior gives no reference.
    """
    q = _validate_quad(frame_quad, "edge_station_bounds")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    H, W = gray.shape
    M = cv2.getPerspectiveTransform(
        q.astype(np.float32),
        np.array([[0, 0], [99, 0], [99, 99], [0, 99]], np.float32))
    inner = cv2.warpPerspective(gray, M, (100, 100))
    level_raw = float(np.percentile(inner[15:85, 15:85], 55))
    if level_raw <= 0:
        return None
    paper_level = 0.82 * level_raw
    ink_level = 0.50 * level_raw
    floor_level = 0.55 * level_raw
    r = max(3, int(0.004 * float(np.hypot(*(q[1] - q[0])))))
    geo = _edge_geometry(q)
    perp = _perp_sizes(q)
    ink_all, ext_all, oob_all = [], [], []
    for i, (a, b, n, _mid) in enumerate(geo):
        max_d = float(min(0.17 * perp[i], 0.11 * min(H, W)))
        orn_window = float(min(0.095 * perp[i], 95.0))
        look_ahead = float(min(0.13 * perp[i], 130.0))
        step = 1.5
        ink_arr = np.zeros(samples)
        ext_arr = np.zeros(samples)
        oob_arr = np.zeros(samples, dtype=bool)
        for si, t in enumerate(np.linspace(0.04, 0.96, samples)):
            p0 = a + (b - a) * t
            d_border = max_d
            for comp, lo_b, hi_b in ((0, 0.5, W - 1.5), (1, 0.5, H - 1.5)):
                nc, pc = float(n[comp]), float(p0[comp])
                if abs(nc) < 1e-9:
                    continue
                for bound in (lo_b, hi_b):
                    dd = (bound - pc) / nc
                    if 0 <= dd < d_border:
                        d_border = dd
            d_cap = min(max_d, d_border)
            hit_border = d_border <= max_d

            def probe(d):
                c = p0 + n * (d + r)
                if c[0] < -r or c[1] < -r or c[0] > W - 1 + r \
                        or c[1] > H - 1 + r:
                    return None
                x, y = int(c[0]), int(c[1])
                x0, x1 = max(0, x - r), min(W, x + r + 1)
                y0, y1 = max(0, y - r), min(H, y + r + 1)
                gp = gray[y0:y1, x0:x1]
                sp = sat[y0:y1, x0:x1]
                return {
                    "pf": float(((gp >= paper_level)
                                 & (sp <= 110)).mean()),
                    "df": float((gp < ink_level).mean()),
                    "sf": float(((sp > 130)
                                 & (gp < paper_level)).mean()),
                    "g": float(gray[min(max(y, 0), H - 1),
                                    min(max(x, 0), W - 1)]),
                    "s": float(sat[min(max(y, 0), H - 1),
                                   min(max(x, 0), W - 1)]),
                }

            def plateau_at(ds):
                """3 consecutive mid-tone uniform low-sat points at ds
                (list of depths): returns their median gray or None."""
                gs = []
                for d in ds:
                    pr = probe(d)
                    if pr is None or pr["df"] >= 0.5 or pr["s"] > 110 \
                            or pr["g"] < 0.50 * level_raw \
                            or pr["g"] >= paper_level:
                        return None
                    gs.append(pr["g"])
                med = float(np.median(gs))
                if med < floor_level:
                    return None
                if max(gs) > 1.15 * med or min(gs) < 0.85 * med:
                    return None
                return med

            def scan_ahead(d0):
                """Past a dark/saturated run: ('paper', d) if strict
                paper resumes, ('plateau', d, med) if a shadow-paper
                plateau forms, else None. Resumption must be
                SUSTAINED (3 consecutive paper probes): the pale
                tracery inside a printed emblem's rim can pass one
                paper probe, and calling that resumption under-reads
                the emblem's true height by tens of px. And after a
                resumption, keep watching a further 25px: a printed
                emblem can sit past a bright ring just outside the
                frame band (the ring itself reads as paper), with the
                emblem body beyond it -- face content ends at the
                LAST resumption whose content run is followed by
                sustained paper again."""
                pts = []
                d2 = d0 + step
                while d2 <= min(d0 + look_ahead, d_cap):
                    pr = probe(d2)
                    if pr is None:
                        break
                    pts.append((d2, pr))
                    d2 += step

                def resume_at(idx):
                    """Depth of the first sustained-paper run starting
                    at or after pts[idx], or (None, None)."""
                    run = 0
                    for k in range(idx, len(pts)):
                        if pts[k][1]["pf"] >= 0.6:
                            run += 1
                            if run >= 3:
                                return pts[k - 2][0], k - 2
                        else:
                            run = 0
                    return None, None

                resume, ridx = resume_at(0)
                if resume is not None:
                    # Hunt for further content runs past bright rings.
                    while True:
                        nxt = None
                        for k in range(ridx + 3, len(pts)):
                            if pts[k][0] > resume + 25.0:
                                break
                            pr = pts[k][1]
                            if pr["df"] >= 0.5 or pr["sf"] >= 0.4:
                                nxt = k
                                break
                        if nxt is None:
                            break
                        resume2, ridx2 = resume_at(nxt)
                        if resume2 is None:
                            # A further run with no paper beyond it
                            # is background (the desk past the sheet's
                            # edge), not face content -- it belongs to
                            # the extent side. Ignore it; never let it
                            # poison the resumption already found.
                            break
                        resume, ridx = resume2, ridx2
                    return ("paper", resume, None)
                # No paper resumption: a shadow-paper plateau?
                plat = []
                for d2, pr in pts:
                    if pr["df"] < 0.5 and pr["s"] <= 110 \
                            and 0.50 * level_raw <= pr["g"] < paper_level:
                        plat.append(d2)
                        if len(plat) >= 3:
                            med = plateau_at(plat[-3:])
                            if med is not None:
                                return ("plateau", plat[-3], med)
                    else:
                        plat = []
                return None

            ink_outer = 0.0
            extent = None
            in_paper = False
            run = 0
            misses = 0
            last_good = 0.0
            ref = level_raw
            ref_hist = []
            plat_run = []
            d = 0.0
            while d <= d_cap:
                pr = probe(d)
                if pr is None:
                    extent = last_good if in_paper else max(ink_outer,
                                                            last_good)
                    oob_arr[si] = True
                    break
                if not in_paper:
                    if (pr["df"] >= 0.5 or pr["sf"] >= 0.5) \
                            and ink_outer == 0.0:
                        got = scan_ahead(d)
                        if got is None:
                            extent = max(ink_outer, last_good)
                            break
                        kind, d2, med = got
                        if kind == "paper":
                            ink_outer = d2
                            last_good = d2
                            run = 1
                            ref_hist = []
                            d = d2 + step
                            continue
                        # Plateau past a frame-edge shadow line: the
                        # dark run was the frame's own shadow, and the
                        # plateau is shadowed paper -- not face ink.
                        in_paper = True
                        ref = med
                        ref_hist = [med]
                        last_good = d2
                        d = d2 + step
                        continue
                    if pr["pf"] >= 0.6:
                        run += 1
                        last_good = d
                        ref_hist.append(pr["g"])
                        if run >= 3:
                            in_paper = True
                            ref = float(np.median(ref_hist[-5:]))
                        d += step
                        continue
                    # Mid-tone plateau candidate (shadowed margin
                    # touching the frame directly).
                    if pr["df"] < 0.5 and pr["s"] <= 110 \
                            and 0.50 * level_raw <= pr["g"] < paper_level:
                        plat_run.append((d, pr["g"]))
                        if len(plat_run) >= 3:
                            med = plateau_at([p[0] for p in plat_run[-3:]])
                            if med is not None:
                                in_paper = True
                                ref = med
                                ref_hist = [med]
                                last_good = plat_run[-3][0]
                                d += step
                                continue
                    else:
                        plat_run = []
                    if d > orn_window:
                        extent = max(ink_outer, last_good)
                        break
                    d += step
                    continue
                # In paper: strict test, or adaptive follow of the
                # local (possibly shadowed) reference.
                ok = pr["pf"] >= 0.6 or (
                    pr["g"] >= 0.78 * ref and pr["g"] >= floor_level
                    and pr["s"] <= 110 and pr["pf"] >= 0.25)
                if ok:
                    last_good = d
                    misses = 0
                    ref_hist.append(pr["g"])
                    ref = float(np.median(ref_hist[-5:]))
                    d += step
                    continue
                misses += 1
                if misses >= 2:
                    extent = last_good
                    break
                d += step
            if extent is None:
                extent = d_cap
                if hit_border:
                    oob_arr[si] = True
            ink_arr[si] = ink_outer
            ext_arr[si] = max(0.0, float(extent))
        ink_all.append(ink_arr)
        ext_all.append(ext_arr)
        oob_all.append(oob_arr)
    return ink_all, ext_all, oob_all, paper_level


def _robustify_bounds(ink_all, ext_all, oob_all, grace=12.0):
    """De-speckle the station bounds without touching real features.

    Neighbour logic, deliberately conservative in both directions:
    an ink value is trimmed only when it spikes above BOTH neighbours
    by more than `grace` (a one-station speck -- a smooth content
    hill like the emblem, several stations wide with high neighbours,
    is never trimmed: cutting face content is the one unforgivable
    error); an extent value is lifted only when it dips below BOTH
    neighbours by more than `grace` (a one-station pit -- a genuine
    >= 2-station intrusion keeps its full bite), and then only up to
    the lower neighbour and by at most `grace`."""
    ink2, ext2, oob2 = [], [], []
    for i in range(4):
        k0 = np.asarray(ink_all[i], dtype=np.float64)
        e0 = np.asarray(ext_all[i], dtype=np.float64)
        k = k0.copy()
        e = e0.copy()
        for s in range(len(k0)):
            kl = k0[s - 1] if s > 0 else k0[s]
            kr = k0[s + 1] if s < len(k0) - 1 else k0[s]
            if k0[s] - max(kl, kr) > grace:
                k[s] = max(kl, kr) + grace
            el = e0[s - 1] if s > 0 else e0[s]
            er = e0[s + 1] if s < len(e0) - 1 else e0[s]
            if min(el, er) - e0[s] > grace:
                e[s] = min(e0[s] + grace, min(el, er))
        ink2.append(k)
        ext2.append(e)
        oob2.append(np.asarray(oob_all[i], dtype=bool) & (e <= e0))
    return ink2, ext2, oob2


def _envelope_solve(bgr, frame, bounds, out_wh, m_lr, m_tb, m_min,
                    graze=0.0, slack=0.0):
    ink_all, ext_all, oob_all, _lvl = bounds
    W, H = float(out_wh[0]), float(out_wh[1])
    canvas = np.array([[0, 0], [W, 0], [W, H], [0, H]], dtype=np.float64)
    geo = _edge_geometry(frame)
    img_h, img_w = bgr.shape[:2]

    def need_allow(i, si):
        # graze/slack are the disc sampler's own measurement tolerance
        # (disc radius ~6-7px straddles every boundary it reads): they
        # are only ever non-zero on the caller's second, last-resort
        # pass. slack never applies to oob stations -- the image
        # border is a hard physical limit, not a measurement.
        need = max(0.0, float(ink_all[i][si]) + 1.5 - graze)
        allow = float(ext_all[i][si]) - (0.0 if oob_all[i][si] else 1.0) \
            + (0.0 if oob_all[i][si] else slack)
        if oob_all[i][si] and need > allow:
            # Photographed flush with the image edge: the framing line
            # may graze the frame's hairline there (the photo itself
            # cut the sheet) rather than leave a black band -- the
            # same trade clamp_quad_to_frame's reserve override makes.
            need = allow
        return need, allow

    hard_bad = 0
    for i in range(4):
        for si in range(len(ink_all[i])):
            nd, al = need_allow(i, si)
            if nd > al + 0.5:
                hard_bad += 1
    if hard_bad:
        return None, {"reason": "station window empty",
                      "hard_bad": int(hard_bad)}

    def build(mlr, mtb):
        rect = np.array([[mlr, mtb], [W - mlr, mtb],
                         [W - mlr, H - mtb], [mlr, H - mtb]],
                        dtype=np.float64)
        hf = cv2.getPerspectiveTransform(frame.astype(np.float32),
                                         rect.astype(np.float32))
        q = cv2.perspectiveTransform(
            canvas.reshape(1, 4, 2).astype(np.float32),
            np.linalg.inv(hf).astype(np.float32)).reshape(4, 2)
        return q.astype(np.float64)

    def pair_viol(quad, edges):
        ink_v = ext_v = 0
        for i in edges:
            a, b, n, _m = geo[i]
            qa, qb = quad[i], quad[(i + 1) % 4]
            for si, t in enumerate(np.linspace(0.04, 0.96,
                                               len(ink_all[i]))):
                p_f = a + (b - a) * t
                p_q = qa + (qb - qa) * t
                dist = float(np.dot(p_q - p_f, n))
                nd, al = need_allow(i, si)
                if dist < nd - 0.25:
                    ink_v += 1
                if dist > al + 0.25:
                    ext_v += 1
        return ink_v, ext_v

    def settle(edges, target, other, is_lr):
        # Station distance (frame edge -> quad edge) GROWS with the
        # output margin: a bigger margin shrinks the frame's target
        # rectangle, so the canvas preimage reaches farther past the
        # frame. Ink violations (quad too close, would cut face
        # content) live at SMALL margins, extent violations (quad past
        # the paper) at LARGE margins: feasible set is a window.
        def mk(m):
            return build(m, other) if is_lr else build(other, m)
        m_hi = target * 1.6
        if pair_viol(mk(m_hi), edges)[0] > 0:
            return None  # even the widest framing cuts the frame
        lo, hi = m_min, m_hi
        for _ in range(14):
            mid = (lo + hi) / 2.0
            if pair_viol(mk(mid), edges)[0] == 0:
                hi = mid
            else:
                lo = mid
        m_ink_min = hi
        if pair_viol(mk(m_min), edges)[1] > 0:
            return None  # even the tightest framing leaves the paper
        lo, hi = m_min, m_hi
        for _ in range(14):
            mid = (lo + hi) / 2.0
            if pair_viol(mk(mid), edges)[1] == 0:
                lo = mid
            else:
                hi = mid
        m_ext_max = lo
        if m_ink_min > m_ext_max + 0.5:
            return None
        if target < m_ink_min:
            return float(m_ink_min)
        if target > m_ext_max:
            return float(m_ext_max)
        return float(target)

    cur_lr, cur_tb = float(m_lr), float(m_tb)
    quad = None
    for _round in range(2):
        v = settle((0, 2), cur_tb, cur_lr, False)
        if v is None:
            return None, {"reason": "top/bottom pair infeasible"}
        cur_tb = v
        v = settle((3, 1), cur_lr, cur_tb, True)
        if v is None:
            return None, {"reason": "left/right pair infeasible"}
        cur_lr = v
        quad = build(cur_lr, cur_tb)
    ink_v, ext_v = pair_viol(quad, (0, 1, 2, 3))
    for c in quad:  # corner pokes beyond station coverage
        if c[0] < -1.5 or c[1] < -1.5 or c[0] > img_w - 0.5 \
                or c[1] > img_h - 0.5:
            ext_v += 1
    info = {"margins_out": [round(cur_tb, 1), round(cur_lr, 1),
                            round(cur_tb, 1), round(cur_lr, 1)],
            "violations": [int(ink_v), int(ext_v)],
            "min_extent": [round(float(np.min(e)), 1) for e in ext_all],
            "max_ink": [round(float(np.max(k)), 1) for k in ink_all]}
    if ink_v or ext_v:
        return None, info
    return quad, info


def envelope_quad(bgr, frame_quad, out_wh, m_lr=250.0, m_tb=370.0,
                  m_min=12.0, extra_tiers=()):
    """Straight-line envelope framing around the decorative frame.

    定案口径 (2026-11-11): the final framing is the straight-line
    envelope of the frame's four edges -- the frame lands in the
    output as a perfect rectangle with a thin, even line of paper
    white around it: left margin == right margin, top == bottom, and
    each margin constant along its edge (the source quad is the
    homography preimage of the canvas under the frame->centered-
    rectangle map). The paper's curled outline is never followed.

    Each margin pair starts at a generous target (m_lr / m_tb output
    px, about 21mm / 31mm on a 300dpi A4 canvas) and settles inside
    its feasible window from edge_station_bounds: the low edge is set
    by the frame's bowed ink and face content riding on it (never cut
    the licence face), the high edge by the paper's safe extent
    (never sample desk, clutter, intrusion or out-of-frame black).

    Zero-fabrication rule (2026-11-11 18:20, ying, after the padded
    envelope was rejected on sight): every output pixel must be a
    resampling of an original pixel. There is NO padding path: when a
    side was photographed flush with the image border and the real
    paper there cannot support the pair margin its partner edge's
    content demands, the pair window is simply empty in real pixels
    and this returns None -- the caller then falls back to a real,
    possibly asymmetric framing. Symmetric margins are the target on
    a normally-shot flat sheet, never a hard requirement, and never
    bought with invented paper. Returns (quad, info) in ORIGINAL
    image coordinates, or (None, info) when no feasible window
    exists.
    """
    frame = np.asarray(frame_quad, dtype=np.float64).reshape(4, 2)
    bounds = edge_station_bounds(bgr, frame)
    if bounds is None:
        return None, {"reason": "no paper reference"}
    ink_all, ext_all, oob_all = _robustify_bounds(*bounds[:3])
    _lvl = bounds[3]
    bounds_r = (ink_all, ext_all, oob_all, _lvl)
    quad, info = _envelope_solve(bgr, frame, bounds_r, out_wh, m_lr,
                                 m_tb, m_min)
    for gz, sl in [(8.0, 6.0)] + list(extra_tiers):
        if quad is not None:
            break
        quad, info = _envelope_solve(bgr, frame, bounds_r, out_wh,
                                     m_lr, m_tb, m_min, graze=gz,
                                     slack=sl)
        if quad is not None:
            info["tolerance"] = [gz, sl]
    return quad, info


def dark_border_fraction(warped_bgr, strip_frac=0.004, dark=None, rel=0.45):
    """Fraction of dark pixels in the outer strips of a warped output.
    Returns dict per side; a healthy license output is ~0 everywhere.

    Threshold is RELATIVE to the paper, not an absolute gray level (fixed
    2026-11-10): paper_level = median gray of the central 60% of the warp,
    dark_thresh = clip(rel * paper_level, 30, 110). An absolute 70 misfires
    both ways -- deep shadow on paper reads as a black border, while a
    dark-grey desk (~80-100) that visibly frames the output reads as clean.
    Pass dark=<number> to force the old absolute behaviour.
    """
    gray = cv2.cvtColor(warped_bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    if dark is None:
        cy0, cy1 = int(H * 0.2), int(H * 0.8)
        cx0, cx1 = int(W * 0.2), int(W * 0.8)
        paper_level = float(np.median(gray[cy0:cy1, cx0:cx1]))
        dark = float(np.clip(rel * paper_level, 30, 110))
    s = max(2, int(strip_frac * min(W, H)))
    sides = {
        "top": gray[:s, :], "bottom": gray[H - s:, :],
        "left": gray[:, :s], "right": gray[:, W - s:],
    }
    out = {k: float((v < dark).mean()) for k, v in sides.items()}
    out["_dark_thresh"] = dark
    return out
