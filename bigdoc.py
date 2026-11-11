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
        if paper is None:
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
