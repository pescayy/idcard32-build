# Full-edge line refinement of a detector frame quad (2026-11-14,
# "corner-to-edge" stage, wired into lic_pipeline between the SCRFD
# candidate and the paper-edge expansion).
#
# Why: the pipeline's geometry model is four straight edge LINES with
# corners as their intersections (see bigdoc), but the detector hands
# it four corner POINTS, and each point carries its own error -- a
# shadow dragging one corner inward by ~100 px tilts the whole edge
# line, and the warped licence's frame line comes out slanted even
# though the physical frame edge is straight (shadow_top_half: the
# rough top edge line sits 38 px inside the gold line at one end and
# 96 px at the other). A model is only ever a coarse grab (ying's
# rule): its corners may not go straight to the output. This stage
# re-measures each edge along its FULL length on the original pixels
# and replaces the two-point line by a robust fit over ~45 stations,
# so one dragged corner is outvoted by the rest of its own edge.
#
# Evidence per edge: at each station along the rough edge, a profile
# is cast along the edge normal across a corridor centred on the
# rough line. The printed frame is a fragmented tracery band, not a
# solid strip, so the profile is read as ink DENSITY (house ink test
# -- dark or saturated against the interior paper level -- smoothed
# over ~16 px). The band is a density segment of plausible strip
# width; a segment running into the corridor's OUTER end is the
# background half-plane (a dark desk reads as "ink" too, and in the
# first calibration it out-massed the real band and dragged three
# fits onto the paper edge -- such segments are disqualified on
# sight), while a segment running into the INNER end is the band
# merged with the face's own ink, whose outer end is still the
# band's outer boundary. Samples outside the image are UNKNOWN, not
# blank: density is normalised over known samples only, unknown
# samples extend the current run, and a segment whose outer end
# disappears into unknown territory is censored and discarded --
# near a corner the corridor leaves the frame, and counting
# out-of-frame samples as blank saws the background half-plane into
# a fake strip (that, not the desk itself, is what dragged the
# second calibration's fits). The station's evidence point is the
# chosen segment's outer boundary. Bands too pale for the ink test
# (a tan frame dipping the grey level by barely a quarter) yield no
# segment at all: those edges simply keep the rough line, by design.
#
# The boundary read scatters by several px from station to station
# (fragmented tracery: one station catches a stroke poking out, the
# next only the band body), so a bare residual gate would refuse
# every honest fit. Trust is instead established the house way --
# stability: the fit must survive being rebuilt from each half of
# the stations separately (the two half-fits agreeing at the edge
# midpoint and in angle), on top of bigdoc._fit_line_robust's MAD
# rejection (the gilt emblem straddles the top band and pokes a few
# stations outward; they are outvoted, not averaged in).
#
# Per-edge verdicts: an edge is only replaced when its evidence is
# strong -- enough stations, enough fit inliers, low residual, and a
# fitted line that stays near the rough line in angle and offset.
# Any edge failing a gate keeps the rough line through the detector
# corners, exactly as the order demands: no evidence, no movement.
# Corners are then the intersections of adjacent edge lines. The
# rebuilt quad must still clear whole-quad gates (area band,
# convexity, per-corner move cap) or the input quad is returned
# untouched. Geometry only: no pixel value is ever read for anything
# but measurement, and none is written.
#
# Joint rectangle-constrained fit (2026-11-14, second round): the
# per-edge stage judges each edge alone, and its failure mode is
# structural -- on shadow_top_half only the right edge passed its
# gates, the rebuilt quad pivoted on that one edge, and the paper
# corners downstream got WORSE (global corner mean 42.7 -> 43.2,
# the target photo 34.7 -> 46.7). But the four edges are not four
# independent lines: the printed frame is ONE rectangle (the
# electronic licence's standard frame, aspect 2101/1322 = 1.589),
# so its photo image is one homography of that rectangle and the
# edges must agree with each other. The joint stage therefore
# re-fits all four edges at once under the rectangle constraint:
# strong-evidence edges anchor the homography, and a weak edge
# (its stations split between the paper edge and a shadow boundary,
# half-fits 16 degrees apart) is placed by the rectangle the other
# three edges imply instead of being fitted to the shadow. The
# fit is a Levenberg-Marquardt minimisation of robust (Huber)
# point-to-line residuals over the same station evidence the
# per-edge stage collected, plus a weak prior pulling the corners
# toward the detector's (sigma 60 px: a steadier, never a master).
# Adjudication is by the order's rule, on internal evidence only:
# the joint quad's total absolute point-to-line residual over ALL
# station points must beat the SCRFD rough quad's, and must not
# lose to the per-edge stage's own output, or the joint proposal
# is discarded whole (info["joint"]["verdict"] says which gate
# refused it) -- and the surviving quad still has to clear the
# same whole-quad gates (area band, convexity, corner move cap
# against the rough quad) before anything is returned. A
# split-half stability gate was tried on top and rejected on
# measurement (see the note in _joint_fit): homographies rebuilt
# from half the stations diverge by 100+ px even for gold-good
# fits, so the test adjudicates nothing at this evidence noise.
import numpy as np
import cv2

from bigdoc import _validate_quad, _edge_geometry, _perp_sizes, \
    _line, _intersect, _fit_line_robust
from lic_wedgecut import SPEC_FRAME_RECT, SPEC_CANVAS

# The standard frame's aspect, from the electronic licence's spec
# rectangle (single source: lic_wedgecut.SPEC_FRAME_RECT).
_SPEC_ASPECT = (SPEC_FRAME_RECT[2] - SPEC_FRAME_RECT[0]) / \
    (SPEC_FRAME_RECT[3] - SPEC_FRAME_RECT[1])
_DOC_CORNERS = np.array([[0.0, 0.0], [_SPEC_ASPECT, 0.0],
                         [_SPEC_ASPECT, 1.0], [0.0, 1.0]])
_DOC_LINES = (np.array([0.0, 1.0, 0.0]),            # top:    y = 0
              np.array([1.0, 0.0, -_SPEC_ASPECT]),  # right:  x = A
              np.array([0.0, 1.0, -1.0]),           # bottom: y = 1
              np.array([1.0, 0.0, 0.0]))            # left:   x = 0
_JOINT_MIN_POINTS = 40         # total station points to attempt
_JOINT_MIN_EDGE_POINTS = 10    # per-edge floor for a "heard" edge
_JOINT_MIN_EDGES = 2           # edges that must be heard
_HUBER_PX = 8.0                # robust loss knee, source px
_PRIOR_SIGMA_PX = 60.0         # detector-corner prior width
_LM_MAX_ITER = 80

_STATIONS = 45
_T_LO, _T_HI = 0.06, 0.94
_STEP = 1.5                    # profile sampling step, source px
_DENS_WIN = 11                 # density smoothing window (samples)
_DENS_MIN = 0.18               # band-zone density threshold
_ZONE_MIN_W = 18.0             # min band-zone width, source px
_MIN_EVID_FRAC = 0.45          # stations with evidence, of sampled
_RMS_MAX = 8.0                 # fit residual ceiling, source px
_ANGLE_MAX_DEG = 5.0           # fitted vs rough direction
_HALF_SEP_MAX = 6.0            # half-fit separation at midpoint, px
_HALF_ANGLE_MAX_DEG = 2.5      # half-fit mutual angle
_MID_OFF_FRAC = 0.08           # fitted line offset at rough midpoint
_CORNER_OFF_FRAC = 0.12        # rough corners' distance to fitted line
_MOVE_CAP_FRAC = 0.15          # rebuilt corner move, of min adjacent edge


def _interior_level(gray, quad):
    """Interior paper reference, same recipe as edge_station_bounds:
    the quad warped to 100x100, 55th percentile of the inner area."""
    M = cv2.getPerspectiveTransform(
        np.asarray(quad, np.float32),
        np.array([[0, 0], [99, 0], [99, 99], [0, 99]], np.float32))
    inner = cv2.warpPerspective(gray, M, (100, 100))
    return float(np.percentile(inner[15:85, 15:85], 55))


def _zone_outer_offset(ink_row, valid_row, offs):
    """Outer boundary offset of the frame band on one station profile,
    or None. ink_row / valid_row: boolean ink and in-frame flags
    along offs (outward-positive). The band zone is a maximal run of
    smoothed ink density at or above _DENS_MIN and at least
    _ZONE_MIN_W wide; the zone holding the rough line (offset 0)
    wins, else the widest-mass zone whose span comes within 0.8 of
    the corridor half-width of 0. Censoring: density is normalised
    over known (in-frame) samples only; unknown samples extend the
    current run; a run reaching the corridor's outer end, or whose
    confirmed ink ends before unknown samples take over, is the
    background half-plane (whole, or sawn off by the frame edge)
    and is disqualified outright."""
    if len(offs) < _DENS_WIN:
        return None
    ker = np.ones(_DENS_WIN)
    den = np.convolve(valid_row.astype(np.float64), ker, mode="same")
    num = np.convolve((ink_row & valid_row).astype(np.float64), ker,
                      mode="same")
    known = den >= 0.6 * _DENS_WIN
    dens = np.where(known, num / np.maximum(den, 1.0), 0.0)
    above = known & (dens >= _DENS_MIN)
    inrun = above | ~known
    runs = []
    s = None
    for i, v in enumerate(inrun):
        if v and s is None:
            s = i
        if not v and s is not None:
            runs.append((s, i - 1))
            s = None
    if s is not None:
        runs.append((s, len(inrun) - 1))
    hw = float(offs[-1])
    best = None
    for s, e in runs:
        idx = np.nonzero(above[s:e + 1])[0]
        if len(idx) == 0:
            continue                    # pure unknown: no read at all
        e_ab = s + int(idx[-1])
        if e_ab < e:
            continue                    # outer end censored by unknown
        if e >= len(offs) - 1:
            continue                    # background reaches corridor end
        s_ab = s + int(idx[0])
        w = float(offs[e_ab] - offs[s_ab])
        if w < _ZONE_MIN_W:
            continue
        mass = float(dens[s_ab:e_ab + 1].sum())
        contains = offs[s_ab] <= 0.0 <= offs[e_ab]
        dist0 = 0.0 if contains else min(abs(float(offs[s_ab])),
                                         abs(float(offs[e_ab])))
        if dist0 > 0.8 * hw:
            continue
        key = (1 if contains else 0, mass)
        if best is None or key > best[0]:
            best = (key, float(offs[e_ab]))
    return None if best is None else best[1]


def _line_from_fit(fit):
    """(centre, direction) from _fit_line_robust -> (a, b, c) line."""
    c, d, _rms, _n = fit
    nrm = np.array([-d[1], d[0]])
    return np.array([nrm[0], nrm[1], -float(np.dot(nrm, c))]), d


def _edge_evidence(gray, sat, level, a, b, n, hw):
    """Evidence points along one edge: for each station, the band
    zone's outer boundary point. Returns (points Nx2, n_stations)."""
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    H, W = gray.shape
    ts = np.linspace(_T_LO, _T_HI, _STATIONS)
    offs = np.arange(-hw, hw + 1e-6, _STEP)
    base = a[None, :] + ts[:, None] * (b - a)[None, :]
    pts = base[:, None, :] + offs[None, :, None] * n[None, None, :]
    xs = pts[:, :, 0]
    ys = pts[:, :, 1]
    valid = (xs >= 0) & (xs <= W - 1) & (ys >= 0) & (ys <= H - 1)
    gm = cv2.remap(gray, xs.astype(np.float32), ys.astype(np.float32),
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sm = cv2.remap(sat, xs.astype(np.float32), ys.astype(np.float32),
                   cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ink = (gm < 0.50 * level) | (sm > 130.0)
    out = []
    for si in range(len(ts)):
        if float(valid[si].mean()) < 0.6:
            continue            # profile mostly out of frame: no read
        off = _zone_outer_offset(ink[si], valid[si], offs)
        if off is not None:
            out.append(base[si] + n * off)
    return np.array(out, float).reshape(-1, 2), len(ts)


def _h_from_corners(corners):
    """Homography (8-vector, h33 = 1) mapping the document
    rectangle's corners onto the given image corners, exactly."""
    M = cv2.getPerspectiveTransform(
        _DOC_CORNERS.astype(np.float32),
        np.asarray(corners, np.float32).reshape(4, 2))
    return np.array([M[0, 0], M[0, 1], M[0, 2],
                     M[1, 0], M[1, 1], M[1, 2],
                     M[2, 0], M[2, 1]])


def _corners_from_h(h):
    H = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]],
                  [h[6], h[7], 1.0]])
    p = np.concatenate([_DOC_CORNERS, np.ones((4, 1))], axis=1) @ H.T
    return p[:, :2] / p[:, 2:3]


def _quad_cost(quad, pts_list):
    """Total absolute point-to-line residual of a plain quad over
    all station evidence (the joint stage's adjudication ruler --
    the same points for every candidate), plus per-edge sums."""
    q = np.asarray(quad, float)
    total = 0.0
    per = []
    for j in range(4):
        pts = pts_list[j]
        if len(pts) == 0:
            per.append(0.0)
            continue
        l = _line(q[j], q[(j + 1) % 4])
        d = np.abs(pts @ np.array([l[0], l[1]]) + l[2])
        per.append(round(float(d.sum()), 1))
        total += float(d.sum())
    return total, per


def _joint_residuals(h, pts_h, weights, rough):
    H = np.array([[h[0], h[1], h[2]], [h[3], h[4], h[5]],
                  [h[6], h[7], 1.0]])
    try:
        Hinv_T = np.linalg.inv(H).T
    except np.linalg.LinAlgError:
        return None
    rs = []
    for j in range(4):
        if len(pts_h[j]) == 0:
            continue
        l = Hinv_T @ _DOC_LINES[j]
        n = float(np.hypot(l[0], l[1]))
        if n < 1e-12 or not np.isfinite(n):
            return None
        d = (pts_h[j] @ l) / n
        u = np.minimum(1.0, _HUBER_PX / np.maximum(np.abs(d), 1e-9))
        rs.append(np.sqrt(weights[j] * u) * d)
    c = _corners_from_h(h)
    if not np.all(np.isfinite(c)):
        return None
    rs.append(((c - rough) / _PRIOR_SIGMA_PX).ravel())
    return np.concatenate(rs)


def _joint_lm(h, pts_h, weights, rough):
    """Levenberg-Marquardt from h; returns (h, weighted SSR)."""
    r = _joint_residuals(h, pts_h, weights, rough)
    if r is None:
        return None, np.inf
    cost = float(r @ r)
    lam = 1e-3
    for _ in range(_LM_MAX_ITER):
        J = np.empty((len(r), 8))
        ok = True
        for k in range(8):
            hk = h.copy()
            step = 1e-6 * max(1.0, abs(h[k]))
            hk[k] += step
            rk = _joint_residuals(hk, pts_h, weights, rough)
            if rk is None:
                ok = False
                break
            J[:, k] = (rk - r) / step
        if not ok:
            break
        A = J.T @ J
        g = J.T @ r
        improved = False
        rel = 0.0
        for _try in range(12):
            try:
                delta = np.linalg.solve(
                    A + lam * np.diag(np.maximum(np.diag(A), 1e-12)),
                    -g)
            except np.linalg.LinAlgError:
                lam *= 10.0
                continue
            rn = _joint_residuals(h + delta, pts_h, weights, rough)
            if rn is not None:
                cn = float(rn @ rn)
                if cn < cost:
                    rel = (cost - cn) / max(cost, 1e-12)
                    h, r, cost = h + delta, rn, cn
                    lam = max(lam / 3.0, 1e-12)
                    improved = True
                    break
            lam *= 10.0
            if lam > 1e12:
                break
        if not improved or rel < 1e-10:
            break
    return h, cost


def _joint_fit(pts_list, weights, rough, base):
    """Rectangle-constrained joint fit of all four frame edges.
    Returns (quad, info) with quad None when the proposal is
    refused; the refusal gate is named in info["verdict"]."""
    jinfo = {"attempted": False}
    total_pts = int(sum(len(p) for p in pts_list))
    heard = int(sum(1 for p in pts_list
                    if len(p) >= _JOINT_MIN_EDGE_POINTS))
    jinfo["points"] = total_pts
    jinfo["edges_heard"] = heard
    if total_pts < _JOINT_MIN_POINTS or heard < _JOINT_MIN_EDGES:
        jinfo["verdict"] = "insufficient_evidence"
        return None, jinfo
    jinfo["attempted"] = True
    cost_rough, per_rough = _quad_cost(rough, pts_list)
    cost_base, _per_base = _quad_cost(base, pts_list)
    jinfo["cost_rough"] = round(cost_rough, 1)
    jinfo["cost_edge_stage"] = round(cost_base, 1)
    pts_h = [np.concatenate([p, np.ones((len(p), 1))], axis=1)
             if len(p) else np.zeros((0, 3)) for p in pts_list]
    best = None
    for cand in (rough, base):
        try:
            h0 = _h_from_corners(cand)
        except cv2.error:
            continue
        h, c = _joint_lm(h0, pts_h, weights, rough)
        if h is None:
            continue
        corners = _corners_from_h(h)
        if np.all(np.isfinite(corners)) \
                and (best is None or c < best[1]):
            best = (corners, c)
    if best is None:
        jinfo["verdict"] = "fit_failed"
        return None, jinfo
    jq = np.asarray(best[0], float)
    cost_joint, per_joint = _quad_cost(jq, pts_list)
    jinfo["cost_joint"] = round(cost_joint, 1)
    jinfo["per_edge_cost_rough"] = per_rough
    jinfo["per_edge_cost_joint"] = per_joint
    if not cost_joint < cost_rough:
        jinfo["verdict"] = "not_better_than_rough"
        return None, jinfo
    if cost_joint > cost_base:
        jinfo["verdict"] = "beaten_by_edge_stage"
        return None, jinfo
    # Whole-quad gates against the rough quad, same bars as the
    # per-edge stage's rebuild (area band, convexity, move cap).
    def _area(p):
        return 0.5 * abs(float(np.dot(p[:, 0], np.roll(p[:, 1], -1))
                               - np.dot(p[:, 1],
                                        np.roll(p[:, 0], -1))))
    moves = [float(np.hypot(*(jq[i] - rough[i]))) for i in range(4)]
    jinfo["corner_moves_px"] = [round(m, 1) for m in moves]
    reason = None
    if not (0.70 * _area(rough) <= _area(jq) <= 1.40 * _area(rough)):
        reason = "area_out_of_band"
    else:
        signs = []
        for i in range(4):
            e1 = jq[(i + 1) % 4] - jq[i]
            e2 = jq[(i + 2) % 4] - jq[(i + 1) % 4]
            signs.append(float(e1[0] * e2[1] - e1[1] * e2[0]))
        if not (all(s > 0 for s in signs)
                or all(s < 0 for s in signs)):
            reason = "non_convex"
        else:
            for i in range(4):
                cap = _MOVE_CAP_FRAC * min(
                    float(np.hypot(*(rough[i] - rough[(i - 1) % 4]))),
                    float(np.hypot(*(rough[(i + 1) % 4] - rough[i]))))
                if moves[i] > cap:
                    reason = "move_capped"
                    break
    if reason is not None:
        jinfo["verdict"] = reason
        return None, jinfo
    # NOTE (SPEC-FIT round, tried and rejected): a split-half
    # stability gate on the homography -- the house trust test
    # that works per-edge -- was implemented and measured on the
    # four images where the joint fit moved the frame most. The
    # half-fit homographies diverged by 122-492 px at the corners
    # INCLUDING on the one fit the gold set verifies as good
    # (shadow_purple, frame error 48.6 -> 21.9): at this evidence
    # noise (station reads scatter +/-20 px), a homography rebuilt
    # from half the stations is ill-conditioned in a way a single
    # line fit is not, so the test refuses everything and
    # adjudicates nothing. It is deliberately NOT in the gate
    # chain; the numbers are in SPEC-FIT-RUN.md for the next round.
    jinfo["verdict"] = "applied"
    return jq, jinfo


def _edge_weight(verdict):
    """Fit weight of one edge's station points in the joint fit,
    from the per-edge stage's own trust grading: a refined edge
    anchors at full weight; an edge whose fit was refused but
    which still produced a full station set is heard at reduced
    weight (the rectangle decides where it lands); sparse
    evidence barely speaks."""
    if verdict.get("verdict") == "refined":
        return 1.0
    if verdict.get("evidence", 0) >= max(10, int(_MIN_EVID_FRAC
                                                 * _STATIONS)):
        return 0.6
    return 0.35


def refine_frame_edges(bgr, quad, edge_stage=True, joint_stage=True):
    """Refine a detector frame quad edge-by-edge. Returns (quad,
    info); the input quad is returned unchanged (the same values)
    unless the rebuilt quad clears every gate. info["edges"][j] is
    the per-edge verdict: "refined" with fit statistics, or "kept_*"
    naming the gate that refused the fit.

    Stage switches: edge_stage gates the per-edge rebuild (v1),
    joint_stage gates the rectangle-constrained joint fit. Either
    may be turned off independently; with both off the rough quad
    is returned untouched and no evidence is gathered. The library
    defaults keep both stages available; the production pipeline
    pins its own delivery configuration (see lic_pipeline's
    EDGE_REFINE_* constants)."""
    try:
        q = _validate_quad(quad, "refine_frame_edges")
    except ValueError:
        return np.asarray(quad, float), {"applied": False,
                                         "reason": "input_insane"}
    if not edge_stage and not joint_stage:
        return q, {"applied": False, "reason": "stages_disabled",
                   "edge_stage": "disabled", "joint_stage": "disabled",
                   "edges": []}
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    sat = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)[:, :, 1].astype(np.float32)
    level = _interior_level(gray, q)
    if level <= 0:
        return q, {"applied": False, "reason": "no_interior_level"}
    geo = _edge_geometry(q)
    perp = _perp_sizes(q)
    names = ("top", "right", "bottom", "left")
    lines = []
    edge_info = []
    pts_list = []
    any_refined = False
    for j, (a, b, n, mid) in enumerate(geo):
        rough_line = _line(a, b)
        verdict = {"edge": names[j]}
        hw = float(min(140.0, max(70.0, 0.10 * perp[j])))
        pts, n_st = _edge_evidence(gray, sat, level, a, b, n, hw)
        pts_list.append(pts)
        verdict["stations"] = int(n_st)
        verdict["evidence"] = int(len(pts))
        line = None
        if len(pts) < max(10, int(_MIN_EVID_FRAC * n_st)):
            verdict["verdict"] = "kept_few_stations"
        else:
            fit = _fit_line_robust(pts)
            if fit is None:
                verdict["verdict"] = "kept_fit_failed"
            else:
                c, d, rms, n_used = fit
                verdict["inliers"] = int(n_used)
                verdict["rms"] = round(float(rms), 2)
                u = np.asarray(b, float) - np.asarray(a, float)
                u = u / max(1e-9, float(np.hypot(*u)))
                cosang = abs(float(np.dot(d, u)))
                ang = float(np.degrees(np.arccos(min(1.0, cosang))))
                verdict["angle_deg"] = round(ang, 2)
                cand, _d = _line_from_fit(fit)
                mid_off = abs(float(cand[0] * mid[0] + cand[1] * mid[1]
                                    + cand[2]))
                verdict["mid_offset_px"] = round(mid_off, 1)
                corner_off = max(
                    abs(float(cand[0] * p[0] + cand[1] * p[1]
                              + cand[2])) for p in (a, b))
                verdict["corner_offset_px"] = round(corner_off, 1)
                # Stability: rebuild the fit from each half of the
                # stations (points arrive in station order). A real
                # straight band gives the same line from either half;
                # a read drifting along the edge (a shadow boundary
                # walking across the corridor, a curl) does not.
                half = len(pts) // 2
                f1 = _fit_line_robust(pts[:half]) if half >= 6 else None
                f2 = _fit_line_robust(pts[half:]) \
                    if len(pts) - half >= 6 else None
                unstable = f1 is None or f2 is None
                if not unstable:
                    l1, d1 = _line_from_fit(f1)
                    l2, d2 = _line_from_fit(f2)
                    hang = float(np.degrees(np.arccos(min(
                        1.0, abs(float(np.dot(d1, d2)))))))
                    p1 = mid - (l1[0] * mid[0] + l1[1] * mid[1]
                                + l1[2]) * np.array([l1[0], l1[1]])
                    p2 = mid - (l2[0] * mid[0] + l2[1] * mid[1]
                                + l2[2]) * np.array([l2[0], l2[1]])
                    hsep = float(np.hypot(*(p1 - p2)))
                    verdict["half_sep_px"] = round(hsep, 1)
                    verdict["half_angle_deg"] = round(hang, 2)
                    unstable = hsep > _HALF_SEP_MAX \
                        or hang > _HALF_ANGLE_MAX_DEG
                if n_used < max(10, 0.5 * len(pts)):
                    verdict["verdict"] = "kept_few_inliers"
                elif rms > _RMS_MAX:
                    verdict["verdict"] = "kept_fit_noisy"
                elif ang > _ANGLE_MAX_DEG:
                    verdict["verdict"] = "kept_angle"
                elif unstable:
                    verdict["verdict"] = "kept_unstable"
                elif mid_off > _MID_OFF_FRAC * perp[j]:
                    verdict["verdict"] = "kept_mid_offset"
                elif corner_off > _CORNER_OFF_FRAC * perp[j]:
                    verdict["verdict"] = "kept_corner_offset"
                else:
                    verdict["verdict"] = "refined"
                    line = cand
                    any_refined = True
        lines.append(rough_line if line is None else line)
        edge_info.append(verdict)
    info = {"applied": False, "edges": edge_info,
            "edge_stage": "on" if edge_stage else "disabled",
            "joint_stage": "on" if joint_stage else "disabled"}
    # Per-edge stage's own candidate: the rebuilt quad when at
    # least one edge was refined and the rebuild clears the
    # whole-quad gates, else the rough quad untouched. With the
    # stage switched off the verdicts above stay on record but
    # nothing is rebuilt.
    base = q
    base_moves = None
    if not edge_stage:
        info["reason"] = "edge_stage_disabled"
    elif not any_refined:
        info["reason"] = "no_edge_refined"
    else:
        corners = []
        parallel = False
        for i in range(4):
            pt = _intersect(lines[(i - 1) % 4], lines[i])
            if pt is None:
                parallel = True
                break
            corners.append(pt)
        if parallel:
            info["reason"] = "parallel_edges"
        else:
            out = np.array(corners, float)
            moves = [float(np.hypot(*(out[i] - q[i])))
                     for i in range(4)]

            def _area(p):
                return 0.5 * abs(float(
                    np.dot(p[:, 0], np.roll(p[:, 1], -1))
                    - np.dot(p[:, 1], np.roll(p[:, 0], -1))))
            reason = None
            if not np.all(np.isfinite(out)):
                reason = "output_nonfinite"
            elif not (0.70 * _area(q) <= _area(out)
                      <= 1.40 * _area(q)):
                reason = "area_out_of_band"
            else:
                signs = []
                for i in range(4):
                    e1 = out[(i + 1) % 4] - out[i]
                    e2 = out[(i + 2) % 4] - out[(i + 1) % 4]
                    signs.append(float(e1[0] * e2[1]
                                       - e1[1] * e2[0]))
                if not (all(s > 0 for s in signs)
                        or all(s < 0 for s in signs)):
                    reason = "non_convex"
                else:
                    for i in range(4):
                        cap = _MOVE_CAP_FRAC * min(
                            float(np.hypot(
                                *(q[i] - q[(i - 1) % 4]))),
                            float(np.hypot(
                                *(q[(i + 1) % 4] - q[i]))))
                        if moves[i] > cap:
                            reason = "move_capped"
                            break
            if reason is not None:
                info["reason"] = reason
            else:
                base = out
                base_moves = moves
    if base_moves is not None:
        info["corner_moves_px"] = [round(m, 1) for m in base_moves]
    # Joint rectangle-constrained stage: it may only replace the
    # per-edge stage's candidate by beating BOTH the rough quad
    # and that candidate on the same internal evidence (see the
    # module header); its verdict is recorded either way.
    if joint_stage:
        weights = [_edge_weight(v) for v in edge_info]
        try:
            jq, jinfo = _joint_fit(pts_list, weights, q, base)
        except Exception as exc:
            jq, jinfo = None, {"attempted": True,
                               "verdict": "error",
                               "error": f"{type(exc).__name__}: {exc}"}
    else:
        jq, jinfo = None, {"attempted": False, "verdict": "disabled"}
    info["joint"] = jinfo
    if jq is not None:
        info["applied"] = True
        info["corner_moves_px"] = jinfo["corner_moves_px"]
        return jq, info
    if base_moves is not None:
        info["applied"] = True
        return base, info
    return q, info
