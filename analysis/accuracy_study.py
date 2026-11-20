"""
Measurement-system study on synthetic back-lit tools (known truth).

Sweeps one influence factor at a time around a baseline (20 px/mm, noise sigma 2, sharp
focus, uniform back light) and records the length and diameter errors over a set of tools
(end mills and 118-degree drills, 3-16 mm diameter). Also measures repeatability: the same
tool imaged 30 times with fresh sensor noise.

Outputs: docs/images/accuracy_study.png and docs/accuracy_study.json
Run:     python analysis/accuracy_study.py
"""
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tool_offsetter.measure import measure_tool  # noqa: E402
from tool_offsetter.synthetic import make_tool_image  # noqa: E402

SIZE, GAUGE = (700, 1000), 900.0
TOOLS = [(d, l, a) for d, l, a in [(3, 15, None), (6, 22, 118), (8, 18, None), (10, 24, 118),
                                    (12, 20, None), (16, 25, 118), (5, 17, 118), (14, 23, None)]]


def run(ppm=20.0, noise=2.0, blur=0.0, gradient=0.0, seed=0):
    eL, eD = [], []
    for i, (d, l, a) in enumerate(TOOLS):
        img, t = make_tool_image(px_per_mm=ppm, diameter_mm=d, length_mm=l, point_angle_deg=a, size=SIZE,
                                 gauge_y_px=GAUGE, axis_x_px=350.4, noise=0, seed=seed + i)
        f = img[..., 0].astype(np.float64)
        if gradient:
            f *= 1 + gradient * (np.linspace(-0.5, 0.5, f.shape[1])[None, :] + np.linspace(-0.25, 0.25, f.shape[0])[:, None])
        if blur:
            f = cv2.GaussianBlur(f, (0, 0), blur)
        f += np.random.default_rng(seed + 100 + i).normal(0, noise, f.shape)
        f = np.clip(f, 0, 255).astype(np.uint8)
        m = measure_tool(f, ppm, GAUGE, band_mm=(1.0, 6.0))
        eL.append(m.length_mm - l)
        eD.append(m.diameter_mm - d)
    eL, eD = np.array(eL) * 1000, np.array(eD) * 1000          # micrometres
    return dict(L_max=float(np.abs(eL).max()), L_mean=float(eL.mean()), D_max=float(np.abs(eD).max()), D_mean=float(eD.mean()))


def main():
    res = {}
    res["resolution"] = {p: run(ppm=p) for p in (5, 8, 12, 16, 20, 25, 32)}
    res["noise"] = {n: run(noise=n) for n in (0, 2, 5, 10, 20)}
    res["blur"] = {b: run(blur=b) for b in (0, 0.5, 1, 2, 3, 5)}
    res["gradient"] = {g: run(gradient=g) for g in (0, 0.1, 0.2, 0.4)}

    # repeatability: one tool (drill 10 x 24), 30 images with fresh noise
    Ls, Ds = [], []
    for s in range(30):
        img, _ = make_tool_image(px_per_mm=20, diameter_mm=10, length_mm=24, point_angle_deg=118, size=SIZE,
                                 gauge_y_px=GAUGE, axis_x_px=350.4, noise=2.0, seed=1000 + s)
        m = measure_tool(img, 20, GAUGE, band_mm=(1.0, 6.0))
        Ls.append(m.length_mm); Ds.append(m.diameter_mm)
    res["repeatability_um"] = {"L_std": float(np.std(Ls, ddof=1) * 1000), "D_std": float(np.std(Ds, ddof=1) * 1000),
                               "L_bias": float((np.mean(Ls) - 24) * 1000), "D_bias": float((np.mean(Ds) - 10) * 1000)}

    (ROOT / "docs").mkdir(exist_ok=True)
    (ROOT / "docs" / "accuracy_study.json").write_text(json.dumps(res, indent=2))

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(2, 2, figsize=(11, 7.5))
    panels = [("resolution", "Camera resolution (px/mm)", "max"), ("noise", "Sensor noise (grey-level sigma)", "max"),
              ("blur", "Defocus blur (Gaussian sigma, px)", "mean"), ("gradient", "Back-light non-uniformity (fraction)", "max")]
    for a, (key, label, kind) in zip(ax.flat, panels):
        xs = list(res[key])
        if kind == "max":
            a.plot(xs, [res[key][x]["L_max"] for x in xs], "o-", label="length, max |error|")
            a.plot(xs, [res[key][x]["D_max"] for x in xs], "s-", label="diameter, max |error|")
        else:
            a.plot(xs, [res[key][x]["L_mean"] for x in xs], "o-", label="length, mean error (bias)")
            a.plot(xs, [res[key][x]["D_mean"] for x in xs], "s-", label="diameter, mean error (bias)")
            a.axhline(0, color="gray", lw=0.8)
        a.set_xlabel(label); a.set_ylabel("error (µm)"); a.grid(alpha=0.3); a.legend(fontsize=8)
    r = res["repeatability_um"]
    fig.suptitle(f"Measurement-system study (synthetic, 8 tools per point)  |  repeatability 1σ: "
                 f"L {r['L_std']:.1f} µm, D {r['D_std']:.1f} µm", fontsize=11)
    fig.tight_layout()
    out = ROOT / "docs" / "images" / "accuracy_study.png"
    fig.savefig(out, dpi=110)
    print(json.dumps(res, indent=1))
    print("saved", out)


if __name__ == "__main__":
    main()
