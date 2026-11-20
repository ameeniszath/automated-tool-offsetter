"""
Tests on synthetic back-lit tools of known size (run: python -m pytest tests, or python tests/test_tool_offsetter.py).
"""
import itertools
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tool_offsetter import calibration  # noqa: E402
from tool_offsetter.measure import measure_tool  # noqa: E402
from tool_offsetter.offsets import fanuc_g10, fanuc_program, length_offset  # noqa: E402
from tool_offsetter.synthetic import make_tool_image  # noqa: E402

TOL_MM = 0.03


def _band(dia, point_angle):
    """Diameter band that starts behind the drill point (cone height = r / tan(angle/2))."""
    start = (dia / 2) / np.tan(np.radians(point_angle / 2)) + 0.5 if point_angle else 0.5
    return (start, start + 5.5)


def test_length_and_diameter_accuracy():
    worst = 0.0
    for ppm, dia, length, angle, seed in itertools.product((12.0, 20.0, 31.7), (3.0, 6.0, 10.0, 16.0),
                                                           (25.0, 38.5), (None, 118), (0, 1)):
        if length * ppm > 850:
            continue
        img, t = make_tool_image(px_per_mm=ppm, diameter_mm=dia, length_mm=length, point_angle_deg=angle,
                                 seed=seed, gauge_y_px=880.25, axis_x_px=450.3)
        m = measure_tool(img, ppm, t["gauge_y_px"], band_mm=_band(dia, angle))
        worst = max(worst, abs(m.length_mm - length), abs(m.diameter_mm - dia))
    assert worst < TOL_MM, worst


def test_tip_down_gives_same_result():
    img, t = make_tool_image(px_per_mm=20, diameter_mm=8, length_mm=30)
    up = measure_tool(img, 20, t["gauge_y_px"], band_mm=(1, 6))
    flipped = img[::-1].copy()
    down = measure_tool(flipped, 20, img.shape[0] - t["gauge_y_px"], band_mm=(1, 6), tip="down")
    assert abs(up.length_mm - down.length_mm) < 1e-6 and abs(up.diameter_mm - down.diameter_mm) < 1e-6


def test_encoder_z_is_added():
    img, t = make_tool_image(px_per_mm=20, diameter_mm=8, length_mm=30)
    m = measure_tool(img, 20, t["gauge_y_px"], band_mm=(1, 6), z_camera_mm=112.35)
    assert abs(m.length_mm - (30 + 112.35)) < TOL_MM


def test_gauge_pin_calibration():
    img, t = make_tool_image(px_per_mm=17.3, diameter_mm=10.0, length_mm=30)
    ppm = calibration.from_gauge_pin(img, 10.0, t["gauge_y_px"], band_px=(20, 120))
    assert abs(ppm - 17.3) / 17.3 < 0.002


def test_two_point_calibration_and_file(tmp_path=None):
    assert abs(calibration.from_two_points((0, 0), (300, 400), 25.0) - 20.0) < 1e-12
    folder = Path(tmp_path) if tmp_path else Path(__file__).parent
    f = folder / "_calib_test.json"
    calibration.save(f, 20.0)
    assert calibration.load(f) == 20.0
    f.unlink()


def test_fanuc_g10_lines():
    assert fanuc_g10(12, 123.4567, 10.0) == ["G10 L10 P12 R123.457;", "G10 L12 P12 R5.000;"]
    assert fanuc_g10(3, 50.0, wear=True) == ["G10 L11 P3 R50.000;"]
    assert abs(length_offset(120.0, "touch_off", 584.788) - (-464.788)) < 1e-9
    prog = fanuc_program([(1, 100.0, 10.0), (2, 80.0, 6.0)])
    assert prog.startswith("%\nO9000") and prog.endswith("M30;\n%") and "G10 L12 P2 R3.000;" in prog


def test_point_angle_and_flat_end():
    for angle in (90, 118, 135, 140):
        img, t = make_tool_image(px_per_mm=20, diameter_mm=10, length_mm=30, point_angle_deg=angle)
        m = measure_tool(img, 20, t["gauge_y_px"], band_mm=(1, 6))      # band starts inside the point: auto-shifted
        assert abs(m.point_angle_deg - angle) < 0.5, (angle, m.point_angle_deg)
        assert abs(m.diameter_mm - 10) < TOL_MM and abs(m.length_mm - 30) < TOL_MM
    img, t = make_tool_image(px_per_mm=20, diameter_mm=10, length_mm=30)
    assert measure_tool(img, 20, t["gauge_y_px"], band_mm=(1, 6)).point_angle_deg is None


def test_defocus_does_not_shorten_drills():
    import cv2
    img, t = make_tool_image(px_per_mm=20, diameter_mm=10, length_mm=30, point_angle_deg=118, noise=0)
    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
    m = measure_tool(blurred, 20, t["gauge_y_px"], band_mm=(1, 6))
    assert abs(m.length_mm - 30) < 0.005, m.length_mm                  # apex from fitted cone edges


def test_profile_tracks_the_silhouette():
    img, t = make_tool_image(px_per_mm=20, diameter_mm=10, length_mm=30, point_angle_deg=118)
    m = measure_tool(img, 20, t["gauge_y_px"], band_mm=(1, 6))
    widths = [w for d, w in m.profile if 5 < d < 10]
    assert widths and abs(np.median(widths) - 10) < 0.02


def test_automated_cycle_long_tools():
    from tool_offsetter.automation import SimulatedStation, auto_measure
    for i, (length, dia, angle) in enumerate([(32.0, 6, 118), (120.0, 16, None), (163.25, 12, 135)]):
        r = auto_measure(SimulatedStation(length, dia, angle, px_per_mm=20.0, seed=i), 20.0)
        assert abs(r.length_mm - length) < TOL_MM and abs(r.diameter_mm - dia) < TOL_MM, (length, r)


def test_blank_image_is_rejected():
    blank = np.full((400, 400, 3), 220, np.uint8)
    try:
        measure_tool(blank, 20, 300)
    except ValueError:
        return
    raise AssertionError("blank image should raise ValueError")


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("passed:", name)
