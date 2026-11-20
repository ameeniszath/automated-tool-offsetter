"""
Pixel-to-millimetre calibration.

Two ways to calibrate:
  * two clicked points on a reference of known length (as in the original program);
  * automatically, from the silhouette of a gauge pin (or any cylinder) of known
    diameter held in the holder - more repeatable than clicking.

The calibration file keeps the original program's format: {"pixels_per_metric": value}.
"""
import json
import math
from pathlib import Path

from .measure import measure_tool


def from_two_points(p1, p2, known_mm):
    """Pixels per mm from two image points a known distance apart."""
    if known_mm <= 0:
        raise ValueError("known length must be positive")
    return math.dist(p1, p2) / known_mm


def from_gauge_pin(image, known_diameter_mm, gauge_y_px, band_px=(20, 120), tip="up"):
    """Pixels per mm from the silhouette width of a pin of known diameter.
    band_px is the row band behind the pin's end used for the width (in pixels)."""
    # measure with px_per_mm = 1 so the 'diameter' comes back in pixels
    m = measure_tool(image, 1.0, gauge_y_px, band_mm=band_px, tip=tip)
    return m.diameter_mm / known_diameter_mm


def load(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text()).get("pixels_per_metric")


def save(path, px_per_mm):
    Path(path).write_text(json.dumps({"pixels_per_metric": px_per_mm}))
