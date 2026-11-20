"""
Automated measuring cycle for tools longer than the camera's field of view.

The camera rides on a Z axis whose position is read by a linear encoder. At Z = 0 the
camera's reference row (the centre crosshair) is on the gauge line. The cycle:

  1. SEARCH : step the camera up (by a fraction of the field of view, so views overlap)
              until the tool tip appears inside the image,
  2. CENTRE : move by the remaining offset so the tip sits on the reference row
              (closed loop: re-image and correct, up to a few iterations),
  3. MEASURE: read the encoder and measure in the image:
              L = z_encoder + (reference_row - tip_row) / k,  plus diameter and point angle.

`Station` is the hardware interface (move_to, read_encoder, grab). `SimulatedStation`
implements it by rendering what the camera would see, so the cycle can be developed and
tested before the stages are built. A real station would wrap the stepper driver and
encoder read-out behind the same three methods.
"""
from dataclasses import dataclass, field

import numpy as np

from .measure import measure_tool, tool_mask
from .synthetic import make_tool_image


class SimulatedStation:
    """Camera on a Z axis looking at a back-lit tool of known size."""

    def __init__(self, length_mm, diameter_mm, point_angle_deg=None, px_per_mm=20.0,
                 frame=(640, 480), encoder_resolution_mm=0.001, positioning_error_mm=0.02,
                 noise=2.0, seed=0):
        self.L, self.D, self.angle = length_mm, diameter_mm, point_angle_deg
        self.k, self.frame = px_per_mm, frame
        self.enc_res, self.pos_err = encoder_resolution_mm, positioning_error_mm
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.z = 0.0                      # true camera height (mm)
        self.moves = 0

    @property
    def ref_row(self):
        return self.frame[1] / 2.0

    def move_to(self, z_mm):
        """Stepper move: lands within +/- positioning_error of the target (encoder reads the truth)."""
        self.z = z_mm + self.rng.uniform(-self.pos_err, self.pos_err)
        self.moves += 1

    def read_encoder(self):
        return round(self.z / self.enc_res) * self.enc_res

    def grab(self):
        gauge_row = self.ref_row + self.z * self.k            # gauge line moves down as the camera rises
        img, _ = make_tool_image(px_per_mm=self.k, diameter_mm=self.D, length_mm=self.L,
                                 point_angle_deg=self.angle, size=self.frame, gauge_y_px=gauge_row,
                                 noise=self.noise, seed=int(self.rng.integers(1 << 30)))
        return img


@dataclass
class CycleResult:
    length_mm: float
    diameter_mm: float
    point_angle_deg: float
    z_encoder_mm: float
    moves: int
    log: list = field(default_factory=list, repr=False)   # (phase, z_encoder, image) for animation


def _tip_row(img):
    """Top-most silhouette row, or None if the silhouette touches the top edge (tip not in view)."""
    mask, _ = tool_mask(img if img.ndim == 2 else img[..., 0])
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0 or rows[0] <= 2:
        return None
    return float(rows[0])


def auto_measure(station, px_per_mm, step_fraction=0.75, max_moves=60, centre_tol_px=2.0,
                 band_mm=(1.0, 6.0), keep_frames=False):
    """Run SEARCH -> CENTRE -> MEASURE on a station. Returns a CycleResult.
    step_fraction: search step as a fraction of the field-of-view height (< 1 so views overlap
    and the tip cannot be skipped)."""
    log = []
    h = station.frame[1]
    ref = station.ref_row
    step_mm = step_fraction * h / px_per_mm
    z_target = 0.0
    station.move_to(z_target)

    # 1. SEARCH
    while True:
        img = station.grab()
        tip = _tip_row(img)
        if keep_frames:
            log.append(("search", station.read_encoder(), img))
        if tip is not None:
            break
        if station.moves >= max_moves:
            raise RuntimeError("Tool tip not found - check the tool and the travel limits.")
        z_target += step_mm
        station.move_to(z_target)

    # 2. CENTRE (closed loop on the image)
    for _ in range(5):
        offset_mm = (ref - tip) / px_per_mm
        if abs(ref - tip) <= centre_tol_px:
            break
        z_target = station.read_encoder() + offset_mm
        station.move_to(z_target)
        img = station.grab()
        tip = _tip_row(img)
        if keep_frames:
            log.append(("centre", station.read_encoder(), img))
        if tip is None:            # overshoot: step back down a little and retry
            station.move_to(station.read_encoder() - step_mm / 4)
            img = station.grab()
            tip = _tip_row(img)

    # 3. MEASURE
    z_enc = station.read_encoder()
    m = measure_tool(img, px_per_mm, ref, band_mm=band_mm, z_camera_mm=z_enc)
    if keep_frames:
        log.append(("measure", z_enc, img))
    return CycleResult(m.length_mm, m.diameter_mm, m.point_angle_deg, z_enc, station.moves, log)
