"""Create the synthetic example images and their annotated results used in the README."""
import sys
from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from tool_offsetter.measure import annotate, measure_tool  # noqa: E402
from tool_offsetter.offsets import fanuc_g10  # noqa: E402
from tool_offsetter.synthetic import make_tool_image  # noqa: E402

cases = [
    ("synthetic_drill", dict(px_per_mm=20, diameter_mm=10, length_mm=36.0, point_angle_deg=118, gauge_y_px=880), (3.5, 9.0), 12),
    ("synthetic_endmill", dict(px_per_mm=20, diameter_mm=16, length_mm=30.0, gauge_y_px=880), (1.0, 8.0), 5),
]
for name, kw, band, tool in cases:
    img, truth = make_tool_image(**kw)
    cv2.imwrite(str(HERE / f"{name}.png"), img)
    m = measure_tool(img, kw["px_per_mm"], kw["gauge_y_px"], band_mm=band)
    cv2.imwrite(str(HERE / f"{name}_result.jpg"), annotate(img, m, tool), [cv2.IMWRITE_JPEG_QUALITY, 88])
    print(f"{name}: true L {truth['length_mm']:.3f} / D {truth['diameter_mm']:.3f} -> "
          f"measured L {m.length_mm:.3f} / D {m.diameter_mm:.3f}   {' '.join(fanuc_g10(tool, m.length_mm, m.diameter_mm))}")
