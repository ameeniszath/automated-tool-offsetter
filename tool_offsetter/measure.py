"""
Tool length and diameter from a back-lit silhouette.

With a back light behind the tool, the tool appears as a dark silhouette on a bright
background. This module:

1. segments the silhouette (Otsu threshold + largest dark connected region),
2. finds the tool tip (the silhouette point furthest from the holder) with sub-pixel
   interpolation along the tool axis,
3. measures tool length from the gauge line (the holder's reference face, or the
   camera's reference row when the camera is moved by an encoder-read Z axis),
4. measures the diameter as the silhouette envelope width in a band behind the tip,
   using sub-pixel edge interpolation on every row of the band.

Image coordinates: x to the right, y downwards, pixel centres at (i + 0.5).
"""
from dataclasses import dataclass, asdict, field

import cv2
import numpy as np


@dataclass
class ToolMeasurement:
    length_mm: float          # gauge line -> tool tip
    diameter_mm: float        # envelope diameter in the measuring band
    tip_x_px: float
    tip_y_px: float
    gauge_y_px: float
    axis_x_px: float
    band_y_px: tuple          # (top, bottom) rows used for the diameter
    px_per_mm: float
    z_camera_mm: float = 0.0  # encoder position of the camera relative to its reference
    point_angle_deg: float = None   # included point angle (drills); None for flat-ended tools
    profile: list = field(default_factory=list, repr=False)  # (distance from tip mm, width mm)

    def as_dict(self, with_profile=False):
        d = asdict(self)
        if not with_profile:
            d.pop("profile")
        return d


def tool_mask(gray, threshold=None):
    """Boolean mask of the largest dark region (the tool) and the threshold used."""
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    if threshold is None:
        threshold, _ = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = (blur < threshold).astype(np.uint8)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    if n <= 1:
        raise ValueError("No dark silhouette found - check the back light and exposure.")
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == k, float(threshold)


def _crossing(profile, start, level, step):
    """Sub-pixel position where `profile` first drops below `level`, scanning from
    `start` in direction `step` (+1 / -1). Returns a coordinate in pixel-centre units."""
    i = start
    n = len(profile)
    while 0 <= i < n and profile[i] >= level:
        i += step
    if not 0 <= i < n:
        return None
    j = i - step                                   # last bright sample
    if not 0 <= j < n:
        return float(i) + 0.5
    g_bright, g_dark = float(profile[j]), float(profile[i])
    frac = (g_bright - level) / (g_bright - g_dark) if g_bright != g_dark else 0.5
    return j + 0.5 + step * frac


def measure_tool(image, px_per_mm, gauge_y_px, band_mm=(1.0, 8.0), tip="up",
                 z_camera_mm=0.0, threshold=None):
    """
    Measure tool length and diameter.

    image       : BGR or grayscale image of the back-lit tool
    px_per_mm   : calibration (pixels per millimetre)
    gauge_y_px  : image row of the gauge line (holder reference face, or the camera's
                  reference row when the camera position comes from an encoder)
    band_mm     : (start, end) distance behind the tip, in mm, used for the diameter.
                  Keep it on the cutting part, clear of the point and of the collet nut.
    tip         : "up" if the tip points to the top of the image, "down" otherwise
    z_camera_mm : camera Z (encoder) relative to the position where the reference row
                  coincided with the gauge line; added to the in-image length
    """
    if px_per_mm is None or px_per_mm <= 0:
        raise ValueError("Calibrate first: px_per_mm must be positive.")
    gray = image if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flip = tip == "down"
    if flip:
        gray = gray[::-1]
        gauge_y_px = gray.shape[0] - gauge_y_px

    mask, _ = tool_mask(gray, threshold)
    rows = np.where(mask.any(axis=1))[0]
    g = gray.astype(np.float64)
    bg = float(np.median(g[~mask]))
    fg = float(np.median(g[mask]))
    level = (bg + fg) / 2.0

    # ---- tip: top-most silhouette row, refined along the column through it ----
    y0 = int(rows[0])
    cols = np.where(mask[y0])[0]
    x_tip_col = int(round(cols.mean()))
    # sample a few rows lower to find a column that is solidly inside the tool
    y_probe = min(y0 + 3, g.shape[0] - 1)
    probe_cols = np.where(mask[y_probe])[0]
    if probe_cols.size:
        x_tip_col = int(round(probe_cols.mean()))
    col = g[:, x_tip_col]
    tip_y = _crossing(col, max(y0 - 3, 0), level, +1)
    if tip_y is None:
        tip_y = y0 + 0.5

    # ---- silhouette profile: sub-pixel left/right edge on every row behind the tip ----
    k = px_per_mm
    gauge_row_now = gauge_y_px + z_camera_mm * k        # where the gauge line is in THIS image
    clip = int(np.floor(min(gauge_row_now - 1, g.shape[0] - 1)))
    scan_end = int(np.floor(min(tip_y + (band_mm[1] + 12.0) * k, clip)))
    prof = {}                                           # row -> (x_left, x_right)
    for y in range(int(np.ceil(tip_y)), scan_end + 1):
        e = _row_edges(g, mask, y, level)
        if e is not None:
            prof[y] = e

    def band_width(top, bottom):
        ws = [prof[y][1] - prof[y][0] for y in range(top, bottom + 1) if y in prof]
        cs = [(prof[y][1] + prof[y][0]) / 2 for y in range(top, bottom + 1) if y in prof]
        return ws, cs

    def envelope(ws):
        """Robust maximum width: smooth along the tool (7 rows) to average out sensor noise,
        then take the 90th percentile so flute lands / margins still count."""
        ws = np.asarray(ws, float)
        if ws.size >= 7:
            ws = np.convolve(ws, np.ones(7) / 7, mode="valid")
        return float(np.percentile(ws, 90))

    top = int(np.ceil(tip_y + band_mm[0] * k))
    bottom = int(np.floor(min(tip_y + band_mm[1] * k, clip)))
    widths, centres = band_width(top, bottom)
    if not widths:
        raise ValueError("No rows in the diameter band - adjust band_mm or the gauge line.")
    width_px = envelope(widths)

    # ---- drill point: rows behind the tip that are still narrower than the body ----
    point_end = None
    for y in sorted(prof):
        if prof[y][1] - prof[y][0] >= 0.95 * width_px:
            point_end = y
            break
    point_angle = None
    if point_end is not None:
        rows_pt = [y for y in prof if y < point_end and 0.15 * width_px < prof[y][1] - prof[y][0] < 0.85 * width_px]
        if len(rows_pt) >= 5:
            ys = np.array(rows_pt, float) + 0.5
            a_l, b_l = np.polyfit(ys, [prof[y][0] for y in rows_pt], 1)
            a_r, b_r = np.polyfit(ys, [prof[y][1] for y in rows_pt], 1)
            point_angle = float(np.degrees(np.arctan(abs(a_l)) + np.arctan(abs(a_r))))
            # Pointed tool: the apex is where the two cone edges meet. Straight edges are not
            # shifted by symmetric defocus blur, whereas the faint apex pixel is - so use the
            # line intersection for the tip (falls back to the intensity tip if fits disagree).
            if abs(a_l - a_r) > 1e-6:
                apex = (b_r - b_l) / (a_l - a_r)
                if abs(apex - tip_y) < 3.0 + 0.05 * width_px:
                    tip_y = float(apex)
        # keep the diameter band clear of the point: shift it if it starts inside the point
        if point_angle is not None and top < point_end + 0.5 * k:
            span = bottom - top
            new_top = int(np.ceil(point_end + 0.5 * k))
            new_bottom = int(min(new_top + span, clip))
            ws, cs = band_width(new_top, new_bottom)
            if ws:
                top, bottom, widths, centres = new_top, new_bottom, ws, cs
                width_px = envelope(widths)
    axis_x = float(np.median(centres))
    length_mm = z_camera_mm + (gauge_y_px - tip_y) / k
    profile = [((y + 0.5 - tip_y) / k, (xr - xl) / k) for y, (xl, xr) in sorted(prof.items())]

    if flip:
        h = gray.shape[0]
        tip_y, gauge_y_px = h - tip_y, h - gauge_y_px      # continuous coordinates
        top, bottom = h - 1 - bottom, h - 1 - top          # integer row indices

    return ToolMeasurement(
        length_mm=float(length_mm), diameter_mm=width_px / k,
        tip_x_px=float(axis_x), tip_y_px=float(tip_y), gauge_y_px=float(gauge_y_px),
        axis_x_px=axis_x, band_y_px=(int(top), int(bottom)), px_per_mm=float(k),
        z_camera_mm=float(z_camera_mm), point_angle_deg=point_angle, profile=profile,
    )


def _row_edges(g, mask, y, level):
    """Sub-pixel (x_left, x_right) of the silhouette on row y, or None."""
    cols = np.where(mask[y])[0]
    if cols.size < 2:
        return None
    row = g[y]
    xl = _crossing(row, max(cols[0] - 3, 0), level, +1)
    xr = _crossing(row, min(cols[-1] + 3, row.size - 1), level, -1)
    if xl is None or xr is None or xr <= xl:
        return None
    return xl, xr


def annotate(image, m, tool_number=None):
    """Draw gauge line, tip, length arrow, diameter band and the readings on a copy."""
    out = image.copy() if image.ndim == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    h, w = out.shape[:2]
    gy, ty, ax = int(round(m.gauge_y_px)), int(round(m.tip_y_px)), int(round(m.axis_x_px))
    half = int(round(m.diameter_mm * m.px_per_mm / 2))
    cv2.line(out, (0, gy), (w, gy), (0, 200, 255), 1)
    ref_label = "gauge line" if m.z_camera_mm == 0 else f"reference row = gauge line + Z {m.z_camera_mm:.3f} mm"
    cv2.putText(out, ref_label, (10, gy - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 150, 255), 1)
    cv2.line(out, (ax - 40, ty), (ax + 40, ty), (0, 0, 255), 1)
    cv2.circle(out, (ax, ty), 4, (0, 0, 255), -1)
    if abs(gy - ty) > 15:
        x_arrow = min(ax + half + 40, w - 10)
        cv2.arrowedLine(out, (x_arrow, gy), (x_arrow, ty), (0, 0, 255), 2, tipLength=0.03)
    t, b = m.band_y_px
    cv2.rectangle(out, (ax - half, t), (ax + half, b), (0, 255, 0), 1)
    label = f"T{tool_number} " if tool_number is not None else ""
    cv2.putText(out, f"{label}L = {m.length_mm:.3f} mm", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.putText(out, f"D = {m.diameter_mm:.3f} mm", (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 160, 0), 2)
    if m.point_angle_deg is not None:
        cv2.putText(out, f"Point = {m.point_angle_deg:.1f} deg", (10, 94), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (160, 0, 160), 2)
    return out
