from __future__ import annotations

import cv2
import numpy as np

SHRINK = 0.35
SHRINK_Y = 0.3
RULE_SPAN = 0.6
RULE_MIN_LEN = 0.1
RULE_THR = 0.3
RULE_INNER_SPAN = 0.4
RULE_CHAIN_DY = 0.4
RULE_CHAIN_DX = 1.5
RULE_GUTTER_DX = 5.0
RULE_WAVE = 3
ROW_DY = 0.8
SINGLE_LINE_STRIP = 2.0
TEXT_PAGE = 0.9
ORPHAN_MAX_LINES = 2
ORPHAN_PITCH = 1.6
ORPHAN_ALIGN = 0.6
ORPHAN_LH = (0.8, 1.25)
INDENT = 0.3
LARGE = 3.0


def hull(lines):
    pts = [p for ln in lines for p in ln.polygon]
    if len(pts) < 3:
        return pts
    h = cv2.convexHull(np.array(pts, dtype=np.float32))
    return [(int(p[0][0]), int(p[0][1])) for p in h]


def _lh(ln) -> float:
    return max(1.0, float(ln.heights[0] + ln.heights[1]))


def _baselines(r):
    return [float(np.mean([p[1] for p in ln.baseline])) for ln in r.lines] or [0.0]


def _line_band(ln):
    cy = float(np.mean([p[1] for p in ln.baseline]))
    return cy - float(ln.heights[0]), cy + float(ln.heights[1])


def _bbox(r):
    pts = [p for ln in r.lines for p in ln.polygon] or list(r.polygon)
    if not pts:
        return (0.0, 0.0, 0.0, 0.0)
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    return (float(min(xs)), float(min(ys)), float(max(xs)), float(max(ys)))


def _gaps(intervals, lo, hi):
    out = []
    cur = lo
    for a, b in sorted(intervals):
        if a > cur:
            out.append((cur, a))
        cur = max(cur, b)
    return out


def _segments_to_rules(segs, lh, chain_dx=RULE_CHAIN_DX):
    segs = sorted(segs, key=lambda v: v[1])
    chains = []
    for y, x0, x1 in segs:
        for c in chains:
            if abs(c[0] - y) <= RULE_CHAIN_DY * lh and x0 - c[2] <= chain_dx * lh:
                c[0] = 0.5 * (c[0] + y)
                c[2] = max(c[2], x1)
                break
        else:
            chains.append([y, x0, x1])
    return [tuple(c) for c in chains]


def _long_runs(binary, lh):
    W = binary.shape[1]
    k = max(15, int(RULE_MIN_LEN * W))
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (k, 1)))
    n, _, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    return [(y + h / 2.0, float(x), float(x + w)) for x, y, w, h in (stats[i, :4] for i in range(1, n))
            if w >= k and h <= max(3.0, 0.6 * lh)]


def _profile_rule(binary, labels, component, stats, lh, min_length):
    x, y, width, height = map(int, stats[:4])
    if width < min_length or height > max(3.0 * lh, 0.06 * width):
        return None
    support = labels[y:y + height, x:x + width] == component
    counts = support.sum(axis=0)
    occupied = counts > 0
    if occupied.mean() < 0.95:
        return None
    columns = np.flatnonzero(occupied)
    centers = (support * np.arange(height)[:, None]).sum(axis=0)[occupied] / counts[occupied]
    slope, intercept = np.polyfit(columns, centers, 1)
    residual = centers - (slope * columns + intercept)
    if abs(slope) > 0.04 or np.quantile(np.abs(residual), 0.95) > max(2.0, 0.25 * lh):
        return None
    if np.quantile(np.abs(np.diff(centers)), 0.95) > max(1.0, 0.15 * lh):
        return None
    radius = max(3, int(np.ceil(0.6 * lh)))
    offsets = np.arange(-radius, radius + 1)[:, None]
    rows = np.rint(y + centers).astype(int)[None, :] + offsets
    inside = (rows >= 0) & (rows < binary.shape[0])
    ink = binary[np.clip(rows, 0, binary.shape[0] - 1), x + columns] * inside
    thickness = ink.sum(axis=0)
    if np.quantile(thickness, 0.95) > max(3.0, 0.6 * lh):
        return None
    if np.mean(thickness > max(3.0, 0.6 * lh)) > 0.025:
        return None
    return ((float(y + np.median(centers) + 0.5), float(x), float(x + width)),
            float(slope), float(y + intercept - slope * x))


def _join_profile_runs(binary, profiles, lh, chain_dx, local_length):
    chains = []
    max_gap = chain_dx * lh
    radius = max(2, int(np.ceil(0.2 * lh)))
    for rule, slope, intercept in sorted(profiles, key=lambda item: item[0][1]):
        for j, (old, old_slope, old_intercept) in enumerate(chains):
            gap = rule[1] - old[2]
            if gap < 0 or gap > max_gap + 2 * local_length or abs(slope - old_slope) > 0.01:
                continue
            midpoint = 0.5 * (old[2] + rule[1])
            if abs((slope - old_slope) * midpoint + intercept - old_intercept) > max(2.0, 0.2 * lh):
                continue
            old_width, width = old[2] - old[1], rule[2] - rule[1]
            combined_slope = (old_slope * old_width + slope * width) / (old_width + width)
            combined_intercept = (old_intercept * old_width + intercept * width) / (old_width + width)
            columns = np.arange(int(old[2]), int(rule[1]))
            if len(columns):
                rows = np.rint(combined_slope * columns + combined_intercept).astype(int)
                rows = rows[None, :] + np.arange(-radius, radius + 1)[:, None]
                inside = (rows >= 0) & (rows < binary.shape[0])
                ink = binary[np.clip(rows, 0, binary.shape[0] - 1), columns] * inside
                supported = ink.any(axis=0)
                if np.count_nonzero(~supported) > max_gap:
                    continue
            x0, x1 = old[1], rule[2]
            center = combined_slope * (x0 + x1 - 1) * 0.5 + combined_intercept + 0.5
            chains[j] = ((float(center), x0, x1), combined_slope, combined_intercept)
            break
        else:
            chains.append((rule, slope, intercept))
    return [rule for rule, _, _ in chains]


def _profile_runs(binary, lh, chain_dx):
    width = binary.shape[1]
    min_length = max(15, int(RULE_MIN_LEN * width))
    local_length = max(15, int(2.0 * lh), min_length // 4)
    opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN,
                             cv2.getStructuringElement(cv2.MORPH_RECT, (local_length, 1)))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(opened, connectivity=8)
    rules = []
    for component in range(1, n):
        rule = _profile_rule(binary, labels, component, stats[component], lh, min_length)
        if rule is not None:
            rules.append(rule)
    return _join_profile_runs(binary, rules, lh, chain_dx, local_length)


def detect_rules(sep_map=None, lh=10.0, page_gray=None, chain_dx=RULE_CHAIN_DX):
    if page_gray is not None:
        g = np.asarray(page_gray)
        if g.dtype != np.uint8:
            g = np.clip(g, 0, 255).astype(np.uint8)
        binary = cv2.adaptiveThreshold(g, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 15)
        segs = _long_runs(binary, lh)
        for rule in _profile_runs(binary, lh, chain_dx):
            if not any(abs(rule[0] - old[0]) <= 1.0 and
                       min(rule[2], old[2]) - max(rule[1], old[1]) >=
                       0.9 * (rule[2] - rule[1]) for old in segs):
                segs.append(rule)
        rules = _segments_to_rules(segs, lh, chain_dx)
        if RULE_WAVE > 1:
            wavy = _long_runs(cv2.dilate(binary, np.ones((RULE_WAVE, 1), np.uint8)), lh)
            rules += [r for r in _segments_to_rules(segs + wavy, lh, chain_dx)
                      if r[2] - r[1] >= RULE_SPAN * g.shape[1]]
        return rules
    if sep_map is None:
        return []
    binary = (np.asarray(sep_map, np.float32) > RULE_THR).astype(np.uint8)
    return _segments_to_rules(_long_runs(binary, lh), lh, chain_dx)


class _Page:
    def __init__(self, regions, rules):
        self.regions = regions
        self.box = [_bbox(r) for r in regions]
        lhs = [_lh(ln) for r in regions for ln in r.lines]
        self.lh = float(np.median(lhs)) if lhs else 10.0
        self.sbox = []
        self.large = set()
        for k, (r, (x0, y0, x1, y1)) in enumerate(zip(regions, self.box)):
            rlh = float(np.median([_lh(ln) for ln in r.lines])) if r.lines else self.lh
            if rlh >= LARGE * self.lh:
                self.large.add(k)
            rlh = rlh if rlh >= LARGE * self.lh else self.lh
            sx = min(SHRINK * self.lh, 0.25 * (x1 - x0))
            sy = min(SHRINK_Y * rlh, 0.25 * (y1 - y0))
            self.sbox.append((x0 + sx, y0 + sy, x1 - sx, y1 - sy))
        self.rules = rules
        self.text_page = self.body_fraction(range(len(regions))) >= TEXT_PAGE

    def extent(self, S):
        b = [self.box[i] for i in S]
        return (min(v[0] for v in b), min(v[1] for v in b), max(v[2] for v in b), max(v[3] for v in b))

    def ygaps(self, S):
        b = [self.sbox[i] for i in S]
        return _gaps([(v[1], v[3]) for v in b], min(v[1] for v in b), max(v[3] for v in b))

    def xgaps(self, S):
        b = [self.sbox[i] for i in S]
        return _gaps([(v[0], v[2]) for v in b], min(v[0] for v in b), max(v[2] for v in b))

    def split_y(self, S, cuts):
        groups = [[] for _ in range(len(cuts) + 1)]
        for i in S:
            cy = 0.5 * (self.sbox[i][1] + self.sbox[i][3])
            groups[sum(cy > c for c in cuts)].append(i)
        return [g for g in groups if g]

    def split_x(self, S, cuts):
        groups = [[] for _ in range(len(cuts) + 1)]
        for i in S:
            cx = 0.5 * (self.sbox[i][0] + self.sbox[i][2])
            groups[sum(cx > c for c in cuts)].append(i)
        return [g for g in groups if g]

    def order(self, S):
        if len(S) <= 1:
            return list(S)
        parts = self.cut(S)
        if parts is None or len(parts) < 2:
            return self.fallback(S)
        return [i for p in parts for i in self.order(p)]

    def cut(self, S):
        X0, Y0, X1, Y1 = self.extent(S)
        yg = self.ygaps(S)
        text = self.text_page
        hard = [ry for ry, rx0, rx1 in self.rules
                if min(rx1, X1) - max(rx0, X0) >= RULE_SPAN * (X1 - X0) and self.cuttable(S, ry, rx0, rx1)]
        parts = self.split_y(S, hard) if hard else []
        if len(parts) > 1:
            return parts

        under = self.under_the_line(S) if text else None
        if under:
            return under

        head = self.masthead_header(S)
        if head:
            return head

        if yg:
            strips = self.split_y(S, [0.5 * (a + b) for a, b in yg])
            groups = [strips[0]]
            k = 1
            while k < len(strips):
                cur, st = groups[-1], strips[k]
                if self.mergeable(cur) and self.mergeable(st) and self.same_columns(cur, st):
                    groups[-1] = cur + st
                    k += 1
                    continue
                j = k
                while j < len(strips) and self.single_row(strips[j]):
                    j += 1
                rows = [i for row in strips[k:j] for i in row]
                if (rows and j < len(strips) and self.mergeable(cur) and self.mergeable(strips[j])
                        and self.same_columns(cur, strips[j], strict=True)
                        and self.within_columns(rows, cur + strips[j])):
                    groups[-1] = cur + rows + strips[j]
                    k = j + 1
                    continue
                groups.append(st)
                k += 1
            if len(groups) > 1:
                return self.rule_first(S, groups)

        xg = self.xgaps(S)
        if not xg:
            return None
        inner = [(ry, rx0, rx1) for ry, rx0, rx1 in self.rules
                 if Y0 + self.lh < ry < Y1 - self.lh and rx1 - rx0 >= RULE_INNER_SPAN * (X1 - X0)]
        free = [g for g in xg if not any(rx0 <= 0.5 * (g[0] + g[1]) <= rx1 for _, rx0, rx1 in inner)]
        return self.rule_first(S, self.split_x(S, [0.5 * (a + b) for a, b in (free or xg)]))

    def masthead_header(self, S):
        X0, Y0, X1, Y1 = self.extent(S)
        if Y0 > self.extent(range(len(self.regions)))[1] + self.lh:
            return None
        base = {i: _baselines(self.regions[i]) for i in S}
        top = [i for i in S if self.box[i][1] <= Y0 + self.lh and self.one_row(i)]
        if len(top) < 2:
            return None
        y_ref = float(np.median([np.mean(base[i]) for i in top]))
        row = [i for i in S if self.one_row(i) and abs(float(np.mean(base[i])) - y_ref) < ROW_DY * self.lh]
        if len(row) < 2:
            return None
        rx0, rx1 = min(self.box[i][0] for i in row), max(self.box[i][2] for i in row)
        if rx1 - rx0 < RULE_SPAN * (X1 - X0):
            return None
        ry0, ry1 = min(self.box[i][1] for i in row), max(self.box[i][3] for i in row)
        if not any(i in self.large and ry1 - self.lh <= self.box[i][1] <= ry1 + 0.5 * (self.box[i][3] - self.box[i][1])
                   for i in S):
            return None
        head = [i for i in S if i not in self.large and self.box[i][1] < ry1 - 0.5 * self.lh]
        rest = [i for i in S if i not in head]
        return [head, rest] if head and rest else None

    def one_row(self, i):
        ys = _baselines(self.regions[i])
        return max(ys) - min(ys) < ROW_DY * self.lh

    def rule_first(self, S, parts):
        if not self.text_page:
            return parts
        X0, Y0, X1, Y1 = self.extent(S)
        rank = {i: k for k, p in enumerate(parts) for i in p}
        for ry, rx0, rx1 in sorted(self.rules):
            if not (Y0 + self.lh < ry < Y1 - self.lh and min(rx1, X1) - max(rx0, X0) >= RULE_SPAN * (X1 - X0)):
                continue
            inside = [i for i in S if rx0 <= 0.5 * (self.sbox[i][0] + self.sbox[i][2]) <= rx1]
            above = [i for i in inside if self.sbox[i][3] <= ry]
            below = [i for i in inside if self.sbox[i][1] >= ry]
            if not any(rank[b] < rank[a] for a in above for b in below):
                continue
            low = [i for i in inside if 0.5 * (self.sbox[i][1] + self.sbox[i][3]) > ry]
            rest = [i for i in S if i not in low]
            if low and rest:
                return [rest, low]
        return parts

    def cuttable(self, S, ry, rx0, rx1):
        cross = self.crossing(S, ry)
        if not cross:
            return True
        if not all(self.small(i) or not self.straddles(i, ry) for i in cross):
            return False
        if self.text_page:
            return True
        PX0, _, PX1, _ = self.extent(range(len(self.regions)))
        if min(rx1, PX1) - max(rx0, PX0) < RULE_SPAN * (PX1 - PX0):
            return False
        below = [i for i in S if rx0 <= 0.5 * (self.sbox[i][0] + self.sbox[i][2]) <= rx1
                 and 0.5 * (self.sbox[i][1] + self.sbox[i][3]) > ry]
        return bool(below) and self.body_fraction(below) >= TEXT_PAGE

    def straddles(self, i, y):
        bands = [_line_band(ln) for ln in self.regions[i].lines]
        return any(b[1] <= y for b in bands) and any(b[0] >= y for b in bands)

    def crossing(self, S, y):
        return [i for i in S if self.sbox[i][1] < y < self.sbox[i][3]]

    def small(self, i):
        return len(self.regions[i].lines) <= ORPHAN_MAX_LINES

    def body_fraction(self, S):
        lhs = [_lh(ln) for i in S for ln in self.regions[i].lines]
        return float(np.mean([ORPHAN_LH[0] * self.lh <= v <= ORPHAN_LH[1] * self.lh for v in lhs])) if lhs else 0.0

    def under_the_line(self, S):
        X0, Y0, X1, Y1 = self.extent(S)
        for ry, rx0, rx1 in sorted(self.rules):
            if not (0.5 * (Y0 + Y1) < ry < Y1 - self.lh and rx1 - rx0 >= RULE_INNER_SPAN * (X1 - X0)):
                continue
            inside = lambda i: rx0 <= 0.5 * (self.sbox[i][0] + self.sbox[i][2]) <= rx1
            cross = self.crossing(S, ry)
            if not any(not inside(i) and not self.small(i) for i in cross):
                continue
            if any(inside(i) and not self.small(i) for i in cross):
                continue
            below = [i for i in S if inside(i) and 0.5 * (self.sbox[i][1] + self.sbox[i][3]) > ry]
            above = [i for i in S if inside(i) and i not in below]
            if not above or not any(rx0 < 0.5 * (a + b) < rx1 for a, b in self.xgaps(above)):
                continue
            rest = [i for i in S if i not in below]
            if below and rest:
                return [rest, below]
        return None

    def same_columns(self, A, B, strict=False):
        ga, gb, gu = self.xgaps(A), self.xgaps(B), self.xgaps(A + B)
        if len(gu) < max(1, min(len(ga), len(gb))):
            return False
        mids = [0.5 * (u[0] + u[1]) for u in gu]
        if strict and not all(any(g[0] <= m <= g[1] for m in mids) for g in min(ga, gb, key=len)):
            return False
        for g in ga + gb:
            if sum(g[0] <= 0.5 * (u[0] + u[1]) <= g[1] for u in gu) > 1:
                return False
        return True

    def single_row(self, S):
        _, y0, _, y1 = self.extent(S)
        if all(len(self.regions[i].lines) <= 1 for i in S) or (y1 - y0) <= SINGLE_LINE_STRIP * self.lh:
            return True
        lines = [ln for i in S for ln in self.regions[i].lines]
        ys = [float(np.mean([p[1] for p in ln.baseline])) for ln in lines]
        return max(ys) - min(ys) < ROW_DY * float(np.median([_lh(ln) for ln in lines]))

    def mergeable(self, S):
        return bool(self.xgaps(S)) and not self.single_row(S)

    def within_columns(self, rows, C):
        mids = [0.5 * (a + b) for a, b in self.xgaps(C)]
        return all(not any(self.sbox[i][0] < m < self.sbox[i][2] for m in mids) for i in rows)

    def fallback(self, S):
        from .postprocess import _order_regions
        idx = {id(self.regions[i]): i for i in S}
        return [idx[id(r)] for r in _order_regions([self.regions[i] for i in S])]


def order_regions(regions, page_w=None, page_h=None, sep_map=None, page_gray=None):
    if len(regions) <= 1:
        return list(regions)
    lhs = [_lh(ln) for r in regions for ln in r.lines]
    lh = float(np.median(lhs)) if lhs else 10.0
    text = _Page(regions, []).text_page
    page = _Page(regions, detect_rules(sep_map, lh, page_gray, RULE_GUTTER_DX if text else RULE_CHAIN_DX))
    return [regions[i] for i in page.order(list(range(len(regions))))]


def _line_stats(r):
    ys = sorted(float(np.mean([p[1] for p in ln.baseline])) for ln in r.lines)
    x0s = [min(p[0] for p in ln.baseline) for ln in r.lines]
    x1s = [max(p[0] for p in ln.baseline) for ln in r.lines]
    lh = float(np.median([_lh(ln) for ln in r.lines]))
    pitch = float(np.median(np.diff(ys))) if len(ys) >= 2 else lh
    return {"ys": ys, "x0": float(min(x0s)), "x0body": float(np.median(x0s)), "x1": float(max(x1s)),
            "lh": lh, "pitch": max(pitch, 0.8 * lh)}


def attach_orphans(regions, rules=()):
    regions = list(regions)
    if len(regions) < 2:
        return regions
    st = [_line_stats(r) if r.lines else None for r in regions]
    order = sorted(range(len(regions)), key=lambda i: st[i]["ys"][0] if st[i] else 0.0)
    absorbed = set()
    for b in order:
        sb = st[b]
        if sb is None or len(regions[b].lines) > ORPHAN_MAX_LINES:
            continue
        best, best_dy = None, None
        for a in range(len(regions)):
            sa = st[a]
            if a == b or a in absorbed or sa is None or len(regions[a].lines) < 2:
                continue
            dy = sb["ys"][0] - sa["ys"][-1]
            if not (0.5 * sa["pitch"] <= dy <= ORPHAN_PITCH * sa["pitch"]):
                continue
            if abs(sb["x0"] - sa["x0body"]) > ORPHAN_ALIGN * sa["lh"]:
                continue
            if sb["x1"] > sa["x1"] + 0.5 * sa["lh"]:
                continue
            if not (ORPHAN_LH[0] * sa["lh"] <= sb["lh"] <= ORPHAN_LH[1] * sa["lh"]):
                continue
            if _rule_between(sa, sb, rules):
                continue
            if best_dy is None or dy < best_dy:
                best, best_dy = a, dy
        if best is None:
            continue
        if _first_line_of_next(b, st, regions, absorbed):
            continue
        regions[best].lines = list(regions[best].lines) + list(regions[b].lines)
        regions[best].polygon = hull(regions[best].lines)
        st[best] = _line_stats(regions[best])
        absorbed.add(b)
    return [r for i, r in enumerate(regions) if i not in absorbed]


def _rule_between(sa, sb, rules):
    for ry, rx0, rx1 in rules:
        if sa["ys"][-1] < ry < sb["ys"][0] and min(rx1, sb["x1"]) - max(rx0, sb["x0"]) > 0:
            return True
    return False


def _first_line_of_next(b, st, regions, absorbed):
    sb = st[b]
    for c in range(len(regions)):
        sc = st[c]
        if c == b or c in absorbed or sc is None:
            continue
        dy = sc["ys"][0] - sb["ys"][-1]
        if not (0 < dy <= ORPHAN_PITCH * sc["pitch"]):
            continue
        if min(sb["x1"], sc["x1"]) - max(sb["x0"], sc["x0"]) < 0.5 * (sb["x1"] - sb["x0"]):
            continue
        if sb["x0"] > sc["x0body"] + INDENT * sc["lh"]:
            return True
    return False


def _page_gray_for(sep_map, page_gray):
    if page_gray is not None and sep_map is not None and page_gray.shape[:2] != sep_map.shape[:2]:
        return None
    return page_gray


def order_page(regions, sep_map, page_gray=None):
    H, W = sep_map.shape[:2] if sep_map is not None else (None, None)
    gray = _page_gray_for(sep_map, page_gray)
    lhs = [_lh(ln) for r in regions for ln in r.lines]
    rules = detect_rules(sep_map, float(np.median(lhs)) if lhs else 10.0, gray)
    regions = attach_orphans(regions, rules)
    return order_regions(regions, W, H, sep_map, gray)


def reorder_page(regions, sep_map, page_gray=None):
    H, W = sep_map.shape[:2]
    return order_regions(regions, W, H, sep_map, _page_gray_for(sep_map, page_gray))
