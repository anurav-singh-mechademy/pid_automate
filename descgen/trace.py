"""Leader-line tracing from an instrument bubble to the object it measures (deterministic, no OCR needed)."""
import math
import numpy as np
import cv2

MM = 300 / 25.4
INK = 160            # leader lines are drawn black; light-grey table rulings (~195) are ignored


def ink_mask(gray):
    return (gray < INK).astype(np.uint8)


def _cov(ink, x, y, ang, L, start=2):
    """Fraction of ink pixels along a ray of length L px from (x, y)."""
    c, s = math.cos(ang), math.sin(ang)
    h, w = ink.shape
    n = hit = 0
    for t in range(start, int(L) + 1):
        xi, yi = int(round(x + c * t)), int(round(y + s * t))
        if not (0 <= xi < w and 0 <= yi < h):
            return 0.0
        n += 1
        hit += ink[yi, xi]
    return hit / max(n, 1)


def _branches(ink, x, y, back, L, thr=0.9, excl=math.radians(50)):
    """Directions of line branches leaving (x, y), excluding the direction we came from."""
    angs = np.radians(np.arange(0, 360, 3))
    cov = np.array([_cov(ink, x, y, a, L) for a in angs])
    ok = cov >= thr
    for i, a in enumerate(angs):
        if abs((a - back + math.pi) % (2 * math.pi) - math.pi) < excl:
            ok[i] = False
    # group consecutive angles into branches (circular)
    out, i, n = [], 0, len(angs)
    if ok.all():
        return []
    start = int(np.argmin(ok))           # rotate so we begin at a non-branch angle
    run = []
    for k in range(n + 1):
        j = (start + k) % n
        if k < n and ok[j]:
            run.append(j)
        elif run:
            best = max(run, key=lambda r: cov[r])
            out.append(float(angs[run[len(run) // 2]]) if cov[run[len(run) // 2]] >= cov[best] - 0.05 else float(angs[best]))
            run = []
    return out


def _exits(ink, cx, cy, r):
    """Angles at which lines leave the bubble: thin dark runs on a ring just outside it; rings further out are tried
    when text beside the bubble (alarm letters) hides the line on the first ring. Returns (angle, ring radius)."""
    return [(a, R) for R in (r + 0.5 * MM, r + 2.0 * MM, r + 3.5 * MM, r + 5.0 * MM) for a in _exits_ring(ink, cx, cy, R)]


def _exits_ring(ink, cx, cy, R):
    angs = np.radians(np.arange(0, 360, 2))
    h, w = ink.shape
    dark = []
    for a in angs:
        xi, yi = int(round(cx + R * math.cos(a))), int(round(cy + R * math.sin(a)))
        dark.append(0 <= xi < w and 0 <= yi < h and ink[max(0, yi - 1):yi + 2, max(0, xi - 1):xi + 2].any())
    dark = np.array(dark)
    if dark.all() or not dark.any():
        return []
    start = int(np.argmin(dark)); out, run = [], []
    for k in range(len(angs) + 1):
        j = (start + k) % len(angs)
        if k < len(angs) and dark[j]:
            run.append(j)
        elif run:
            if len(run) <= 8:                # a line, not a text blob hugging the circle
                out.append(float(angs[run[len(run) // 2]]))
            run = []
    return out


def walk(ink, x, y, ang, max_mm=400, step=4):
    """Follow a straight-segment polyline from (x, y) heading `ang`. Straight continuation is preferred
    (crossing lines are passed through); where the line stops, a single onward branch is a bend and is followed,
    several branches mean the line ran into an object outline (end), none means a free end."""
    L = int(1.3 * MM)
    nx, ny = -math.sin(ang), math.cos(ang)            # snap the start point onto the stroke (perpendicular search)
    h, w = ink.shape
    for d in (0, 1, -1, 2, -2, 3, -3, 4, -4):
        xi, yi = int(round(x + d * nx)), int(round(y + d * ny))
        if 0 <= xi < w and 0 <= yi < h and ink[yi, xi]:
            x, y = x + d * nx, y + d * ny
            break
    pts = [(x, y)]
    seg0 = (x, y)
    total = 0.0
    end = "maxlen"
    while total < max_mm * MM:
        # preferred heading: direction of the current straight segment once it is long enough
        sx, sy = seg0
        if math.hypot(x - sx, y - sy) > 3 * MM:
            ang = math.atan2(y - sy, x - sx)
            tol = range(-3, 4)
        else:
            tol = range(-12, 13, 2)
        best = max(((_cov(ink, x, y, ang + math.radians(d), L), -abs(d), ang + math.radians(d)) for d in tol))
        if best[0] >= 0.9:
            a = best[2]
            x, y = x + step * math.cos(a), y + step * math.sin(a)
            total += step
            pts.append((x, y))
            continue
        # run on to the very end of the stroke before looking for branches (the probe stops ~1 mm short)
        h, w = ink.shape
        for _ in range(int(2 * MM)):
            xi, yi = int(round(x + math.cos(ang))), int(round(y + math.sin(ang)))
            if not (0 <= xi < w and 0 <= yi < h and ink[yi, xi]):
                break
            x, y = x + math.cos(ang), y + math.sin(ang)
        pts.append((x, y))
        br = _branches(ink, x, y, ang + math.pi, L)
        if not br and total > 0:                      # thin diagonal strokes cover less of the probe
            br = _branches(ink, x, y, ang + math.pi, int(2 * MM), thr=0.75)
        if len(br) == 1 and total > 0:
            ang = br[0]; seg0 = (x, y)
            x, y = x + step * math.cos(ang), y + step * math.sin(ang)
            total += step; pts.append((x, y))
            continue
        end = "object" if len(br) >= 2 else "free"
        break
    return pts, end


def trace_bubble(gray, bbox, ink=None):
    """All solid lines leaving a bubble; returns list of dict(pts, end, length_mm, exit_ang)."""
    if ink is None:
        ink = ink_mask(gray)
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    r = ((x2 - x1) + (y2 - y1)) / 4
    ink = ink.copy()
    cv2.circle(ink, (int(cx), int(cy)), int(r + 0.25 * MM), 0, -1)   # white out the bubble (its own text + circle)
    res = []
    for a, R in _exits(ink, cx, cy, r):
        pts, end = walk(ink, cx + R * math.cos(a), cy + R * math.sin(a), a)
        L = sum(math.hypot(pts[i + 1][0] - pts[i][0], pts[i + 1][1] - pts[i][1]) for i in range(len(pts) - 1)) / MM
        res.append(dict(pts=pts, end=end, length_mm=L, exit_ang=math.degrees(a), ring_mm=(R - r) / MM))
    return res
