"""
Measure a tool in a saved image and print its offsets (no camera or GUI needed).

Examples
  python measure_tool.py examples/synthetic_drill.png --px-per-mm 20 --gauge-y 880 --tool 12 --band 9 15
  python measure_tool.py photo.jpg --calib calibration_config.json --gauge-y 1012 --tool 3 --out result.png
  python measure_tool.py photo.jpg --px-per-mm 18.4 --gauge-y 540 --z-camera 112.350   # camera moved up 112.35 mm
"""
import argparse
import json
import sys

import cv2

from tool_offsetter import calibration
from tool_offsetter.measure import annotate, measure_tool
from tool_offsetter.offsets import fanuc_g10


def main():
    ap = argparse.ArgumentParser(description="Tool length and diameter from a back-lit image")
    ap.add_argument("image")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--px-per-mm", type=float, help="calibration in pixels per mm")
    g.add_argument("--calib", help="calibration JSON saved by app.py")
    ap.add_argument("--gauge-y", type=float, required=True, help="image row of the gauge line / camera reference row")
    ap.add_argument("--tool", type=int, default=1)
    ap.add_argument("--band", type=float, nargs=2, default=[1.0, 8.0], metavar=("START_MM", "END_MM"))
    ap.add_argument("--tip", choices=["up", "down"], default="up")
    ap.add_argument("--z-camera", type=float, default=0.0, help="camera Z from the encoder, mm")
    ap.add_argument("--convention", choices=["gauge", "touch_off"], default="gauge")
    ap.add_argument("--gauge-to-zero", type=float, help="for touch_off: gauge line at Z home to work zero, mm")
    ap.add_argument("--out", help="save an annotated image here")
    ap.add_argument("--json", action="store_true", help="print the result as JSON")
    a = ap.parse_args()

    img = cv2.imread(a.image)
    if img is None:
        sys.exit(f"Cannot read {a.image}")
    ppm = a.px_per_mm if a.px_per_mm else calibration.load(a.calib)
    if not ppm:
        sys.exit("No calibration found.")
    try:
        m = measure_tool(img, ppm, a.gauge_y, band_mm=tuple(a.band), tip=a.tip, z_camera_mm=a.z_camera)
    except ValueError as e:
        sys.exit(str(e))
    g10 = fanuc_g10(a.tool, m.length_mm, m.diameter_mm, convention=a.convention,
                    gauge_to_work_zero_mm=a.gauge_to_zero)
    if a.json:
        print(json.dumps({**m.as_dict(), "tool": a.tool, "g10": g10}, indent=2))
    else:
        print(f"Tool T{a.tool}")
        print(f"  length   : {m.length_mm:.3f} mm  (gauge line -> tip)")
        print(f"  diameter : {m.diameter_mm:.3f} mm")
        print(f"  point    : {'flat end' if m.point_angle_deg is None else f'{m.point_angle_deg:.1f} deg'}")
        print("  Fanuc offsets:")
        for line in g10:
            print("    " + line)
    if a.out:
        cv2.imwrite(a.out, annotate(img, m, a.tool))
        print(f"  annotated image: {a.out}")


if __name__ == "__main__":
    main()
