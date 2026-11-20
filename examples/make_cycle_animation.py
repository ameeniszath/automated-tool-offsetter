"""Render the simulated automated measuring cycle as an animated GIF for the README."""
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tool_offsetter.automation import SimulatedStation, auto_measure  # noqa: E402
from tool_offsetter.measure import annotate, measure_tool  # noqa: E402

L, D, ANGLE, K = 96.40, 10.0, 118, 20.0
station = SimulatedStation(L, D, ANGLE, px_per_mm=K, seed=3)
res = auto_measure(station, K, keep_frames=True)

frames = []
for i, (phase, z, img) in enumerate(res.log):
    view = img.copy()
    h, w = view.shape[:2]
    if phase == "measure":
        m = measure_tool(img, K, h / 2, band_mm=(1.0, 6.0), z_camera_mm=z)
        view = annotate(view, m, 7)
    cv2.line(view, (0, h // 2), (w, h // 2), (0, 200, 0), 1)
    cv2.putText(view, "reference row", (w - 150, h // 2 - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 160, 0), 1)
    bar = np.full((70, w, 3), 30, np.uint8)
    cv2.putText(bar, f"{phase.upper():8s}  encoder Z = {z:8.3f} mm", (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    msg = {"search": "stepping up until the tip is in view", "centre": "moving the tip onto the reference row",
           "measure": f"L = Z + offset = {res.length_mm:.3f} mm (true {L:.2f})"}[phase]
    cv2.putText(bar, msg, (12, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 220, 255), 1)
    frame = np.vstack([bar, view])
    frame = cv2.resize(frame, (frame.shape[1] * 3 // 4, frame.shape[0] * 3 // 4), interpolation=cv2.INTER_AREA)
    frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).convert("P", palette=Image.ADAPTIVE, colors=64))

durations = [700] * (len(frames) - 1) + [3500]
out = ROOT / "docs" / "images" / "auto_cycle.gif"
frames[0].save(out, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=True)
print(f"{len(frames)} frames, moves {res.moves}, measured L {res.length_mm:.3f} (true {L}), D {res.diameter_mm:.3f}, "
      f"point {res.point_angle_deg:.1f} -> {out.name} {out.stat().st_size // 1024} KB")
