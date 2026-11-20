"""
Synthetic back-lit tool images with known dimensions, for testing and demos.

Each pixel's grey level is the exact area coverage of the tool, estimated from 8 x 8
sub-samples taken at sub-pixel centres and tested against the (convex) outline with
half-plane inequalities - so edges are anti-aliased like a real camera image and the
geometry is exact (no rasteriser boundary bias). Gaussian noise is added on top.
"""
import cv2
import numpy as np

SS = 8  # sub-samples per pixel along each axis


def _fill_convex(canvas, poly, value, ss=SS):
    """Paint the area coverage of a convex polygon (image coords, any winding) into canvas."""
    poly = np.asarray(poly, float)
    h, w = canvas.shape
    x0, y0 = np.floor(poly.min(0)).astype(int)
    x1, y1 = np.ceil(poly.max(0)).astype(int)
    x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if x0 >= x1 or y0 >= y1:
        return
    area = np.sum(poly[:, 0] * np.roll(poly[:, 1], -1) - np.roll(poly[:, 0], -1) * poly[:, 1])
    sgn = 1 if area > 0 else -1
    # inward unit normals: inside the polygon <=> n . (p - a) >= 0 for every edge
    normals = []
    for (ax, ay), (bx, by) in zip(poly, np.roll(poly, -1, axis=0)):
        n = sgn * np.array([-(by - ay), bx - ax])
        normals.append((n / np.hypot(*n), np.array([ax, ay])))

    def dist(px, py):
        return np.min([n[0] * (px - a[0]) + n[1] * (py - a[1]) for n, a in normals], axis=0)

    Xc, Yc = np.meshgrid(np.arange(x0, x1) + 0.5, np.arange(y0, y1) + 0.5)
    d = dist(Xc, Yc)
    cover = (d >= 0.71).astype(float)                 # pixel entirely inside (half-diagonal 0.707)
    edge = (d > -0.71) & (d < 0.71)                   # only edge pixels need sub-sampling
    if edge.any():
        off = (np.arange(ss) + 0.5) / ss - 0.5
        ox, oy = np.meshgrid(off, off)
        ex, ey = Xc[edge][:, None] + ox.ravel()[None], Yc[edge][:, None] + oy.ravel()[None]
        cover[edge] = (dist(ex, ey) >= 0).mean(axis=1)
    region = canvas[y0:y1, x0:x1]
    region[:] = region * (1 - cover) + value * cover


def make_tool_image(px_per_mm=20.0, diameter_mm=10.0, length_mm=40.0, point_angle_deg=None,
                    holder_width_mm=32.0, size=(900, 1100), gauge_y_px=900.0, axis_x_px=None,
                    background=225, tool_gray=25, noise=2.0, seed=0):
    """Return (BGR image, truth dict). length_mm is gauge line -> tip.
    point_angle_deg=None draws a flat-ended tool (end mill); e.g. 118 draws a drill point."""
    w, h = size
    ax = w / 2 if axis_x_px is None else axis_x_px
    img = np.full((h, w), float(background))

    r = diameter_mm * px_per_mm / 2
    tip_y = gauge_y_px - length_mm * px_per_mm
    if point_angle_deg:
        point_h = r / np.tan(np.radians(point_angle_deg / 2))
        body = [(ax - r, gauge_y_px), (ax - r, tip_y + point_h), (ax, tip_y), (ax + r, tip_y + point_h), (ax + r, gauge_y_px)]
    else:
        body = [(ax - r, gauge_y_px), (ax - r, tip_y), (ax + r, tip_y), (ax + r, gauge_y_px)]
    hw = holder_width_mm * px_per_mm / 2
    nut_top = gauge_y_px - 2.4 * px_per_mm           # collet nut 2.4 mm above the gauge line
    bottom = max(h, gauge_y_px + 3 * px_per_mm) + 1
    holder = [(ax - hw * 0.6, nut_top), (ax + hw * 0.6, nut_top), (ax + hw, gauge_y_px + 3 * px_per_mm),
              (ax + hw, bottom), (ax - hw, bottom), (ax - hw, gauge_y_px + 3 * px_per_mm)]
    _fill_convex(img, holder, tool_gray)
    _fill_convex(img, body, tool_gray)
    img += np.random.default_rng(seed).normal(0, noise, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    truth = dict(length_mm=length_mm, diameter_mm=diameter_mm, tip_y_px=tip_y, gauge_y_px=gauge_y_px,
                 px_per_mm=px_per_mm, point_angle_deg=point_angle_deg)
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR), truth
