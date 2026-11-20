"""
Vision Measurement System + Tool Presetter (interactive).

Based on VisionMeasurementSystem V3.1 (see legacy/). Adds:
  * tool measurement mode (M): tool length from the gauge line and diameter, with the
    Fanuc G10 offset lines printed to the console,
  * gauge-line picking (G),
  * still-image mode (--image) so the program can be used without a camera,
  * configurable camera index, tool number, tip direction and encoder Z.

Keys (live)     : S capture | C calibrate | T camera-shift tracking | D clear calibration | Q quit
Keys (captured) : Y accept | N reject -> 1 contour / 2 multi-point | M measure tool | G set gauge line
                  K snap (multi-point) | Z undo point | R back to live

Run:  python app.py --camera 0            (live camera)
      python app.py --image examples/synthetic_drill.png --tool 12
"""
import argparse
import os
import tkinter as tk
from tkinter import messagebox, simpledialog

import cv2
import numpy as np

from tool_offsetter import calibration
from tool_offsetter.measure import annotate, measure_tool
from tool_offsetter.offsets import fanuc_g10

ap = argparse.ArgumentParser(description="Vision measurement system and tool presetter")
ap.add_argument("--camera", type=int, default=1, help="camera index (default 1, the external USB camera)")
ap.add_argument("--image", help="use a still image instead of a camera")
ap.add_argument("--config", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "calibration_config.json"))
ap.add_argument("--tool", type=int, default=1, help="tool / offset number for the G10 output")
ap.add_argument("--tip", choices=["up", "down"], default="up", help="direction the tool tip points in the image")
ap.add_argument("--band", type=float, nargs=2, default=[1.0, 8.0], metavar=("START_MM", "END_MM"),
                help="diameter band behind the tip, in mm")
ap.add_argument("--z-camera", type=float, default=0.0, help="camera Z from the encoder, mm (0 if the camera is not moved)")
args = ap.parse_args()

# Initialize tkinter root and hide it
root = tk.Tk()
root.withdraw()

CONFIG_PATH = args.config
pixels_per_metric = calibration.load(CONFIG_PATH)
if pixels_per_metric is not None:
    print(f"Loaded calibration: {pixels_per_metric:.4f} pixels/mm")


def save_calibration(val):
    global pixels_per_metric
    pixels_per_metric = val
    try:
        calibration.save(CONFIG_PATH, val)
        print(f"Saved calibration: {pixels_per_metric:.4f} pixels/mm")
    except Exception as e:
        print("Error saving calibration:", e)


state = "LIVE"
captured_frame = None

object_width = 0
object_height = 0

show_manual_menu = False
manual_mode = None

mouse_x = 0
mouse_y = 0

contour_locked = False
selected_contour = None
measurement_printed = False
points = []

calibration_point1 = None
calibration_point2 = None
run_calibration_prompt = False

snap_enabled = False
snap_preview = None
SNAP_RADIUS = 30

# Camera Tracking Variables
tracking_active = False
accumulated_dx = 0.0
accumulated_dy = 0.0
tracking_prev_gray = None
tracking_prev_pts = None

# Tool presetter variables
tool_result = None
gauge_row = None          # None -> use the centre (reference) crosshair row
setting_gauge = False


def mouse_event(event, x, y, flags, param):
    global mouse_x, mouse_y, contour_locked, manual_mode, gauge_row, setting_gauge, tool_result
    global points, calibration_point1, calibration_point2, run_calibration_prompt, state, snap_enabled, snap_preview

    mouse_x = x
    mouse_y = y

    if event == cv2.EVENT_LBUTTONDOWN:
        if setting_gauge:
            gauge_row = y
            setting_gauge = False
            tool_result = None
            print(f"Gauge line set at row {y}")
        elif state == "CALIBRATING":
            if calibration_point1 is None:
                calibration_point1 = (x, y)
            elif calibration_point2 is None:
                calibration_point2 = (x, y)
                run_calibration_prompt = True
        elif manual_mode == "CONTOUR":
            contour_locked = True
        elif manual_mode == "POINT":
            if snap_enabled and snap_preview is not None:
                points.append(snap_preview)
            else:
                points.append((x, y))


if args.image:
    still = cv2.imread(args.image)
    if still is None:
        raise SystemExit(f"Cannot read image {args.image}")
    cap = None
else:
    cap = cv2.VideoCapture(args.camera)

cv2.namedWindow("Vision Measurement System")
cv2.setMouseCallback("Vision Measurement System", mouse_event)

while True:

    if cap is None:
        ret, frame = True, still.copy()
    else:
        ret, frame = cap.read()

    if not ret:
        print("Camera Error")
        break

    # Camera Tracking Processing using Lucas-Kanade Optical Flow
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if tracking_active and state == "LIVE":
        if tracking_prev_gray is not None and tracking_prev_pts is not None and len(tracking_prev_pts) > 0:
            next_pts, status, err = cv2.calcOpticalFlowPyrLK(
                tracking_prev_gray, gray, tracking_prev_pts, None,
                winSize=(21, 21), maxLevel=3,
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
            )

            if next_pts is not None and status is not None:
                good_prev = tracking_prev_pts[status.ravel() == 1].reshape(-1, 2)
                good_next = next_pts[status.ravel() == 1].reshape(-1, 2)

                if len(good_next) >= 5:
                    # Camera moves opposite to feature displacement in the frame
                    dxs = good_prev[:, 0] - good_next[:, 0]
                    dys = good_prev[:, 1] - good_next[:, 1]
                    accumulated_dx += np.median(dxs)
                    accumulated_dy += np.median(dys)
                    tracking_prev_pts = good_next.reshape(-1, 1, 2)
                else:
                    tracking_prev_pts = None
            else:
                tracking_prev_pts = None

        if tracking_prev_pts is None or len(tracking_prev_pts) < 15:
            tracking_prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=100, qualityLevel=0.01, minDistance=10)
        tracking_prev_gray = gray.copy()
    else:
        tracking_prev_gray = None
        tracking_prev_pts = None

    # Check if we need to run the calibration prompt in the main thread
    if run_calibration_prompt:
        run_calibration_prompt = False
        pixel_dist = calibration.from_two_points(calibration_point1, calibration_point2, 1.0)
        try:
            root.deiconify()
            root.focus_force()
            root.withdraw()
            real_size = simpledialog.askfloat(
                "Calibration",
                f"Pixel distance: {pixel_dist:.2f} px\nEnter actual size of reference object (in mm):",
                parent=root,
                minvalue=0.001
            )
            if real_size is not None and real_size > 0:
                ratio = pixel_dist / real_size
                save_calibration(ratio)
                messagebox.showinfo("Calibration Success", f"Calibration successful!\nRatio: {ratio:.4f} pixels/mm")
            else:
                print("Calibration canceled.")
        except Exception as e:
            print("Error during calibration dialog:", e)

        state = "CAPTURED"
        calibration_point1 = None
        calibration_point2 = None

    if state == "LIVE":
        display = frame.copy()

        # Center Reference Crosshair (the camera's reference row = gauge line at encoder zero)
        h, w = display.shape[:2]
        cx = w // 2
        cy = h // 2

        cv2.line(display, (cx - 25, cy), (cx + 25, cy), (0, 255, 0), 1)
        cv2.line(display, (cx, cy - 25), (cx, cy + 25), (0, 255, 0), 1)
        cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

        if tracking_active and tracking_prev_pts is not None:
            for pt in tracking_prev_pts:
                px, py = pt[0]
                cv2.circle(display, (int(px), int(py)), 3, (255, 255, 0), -1)

            dist_px = (accumulated_dx ** 2 + accumulated_dy ** 2) ** 0.5
            if pixels_per_metric is not None:
                track_txt = (f"Cam Shift: DX={accumulated_dx / pixels_per_metric:.2f}mm, "
                             f"DY={accumulated_dy / pixels_per_metric:.2f}mm, Dist={dist_px / pixels_per_metric:.2f}mm")
            else:
                track_txt = f"Cam Shift: DX={accumulated_dx:.1f}px, DY={accumulated_dy:.1f}px, Dist={dist_px:.1f}px"
            cv2.putText(display, track_txt, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(display, "TRACKING ACTIVE (Press T to Reset/Stop)", (20, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        cv2.putText(display, "Press S to Capture | C to Calibrate | T to Track", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    elif state == "CALIBRATING":
        display = captured_frame.copy()
        cv2.putText(display, "CALIBRATING - CLICK 2 POINTS ON REFERENCE", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if calibration_point1 is not None:
            cv2.circle(display, calibration_point1, 5, (0, 0, 255), -1)
        if calibration_point2 is not None:
            cv2.circle(display, calibration_point2, 5, (0, 0, 255), -1)
            cv2.line(display, calibration_point1, calibration_point2, (0, 255, 255), 2)
    else:
        display = captured_frame.copy()
        gray = cv2.cvtColor(display, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blurred, 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Snap preview calculation
        if manual_mode == "POINT":
            snap_preview = None
            if snap_enabled and len(contours) > 0:
                contour = max(contours, key=cv2.contourArea)
                d = np.hypot(contour[:, 0, 0] - mouse_x, contour[:, 0, 1] - mouse_y)
                i = int(np.argmin(d))
                if d[i] <= SNAP_RADIUS:
                    snap_preview = (int(contour[i, 0, 0]), int(contour[i, 0, 1]))

        # ----- Tool presetter result (takes over the view) -----
        if tool_result is not None:
            display = annotate(display, tool_result, args.tool)

        # ----- Auto Contour Mode (Actual Outline + Rotated Bounding Box) -----
        elif len(contours) > 0 and manual_mode is None:
            largest_contour = max(contours, key=cv2.contourArea)
            cv2.drawContours(display, [largest_contour], -1, (0, 255, 0), 2)

            rect = cv2.minAreaRect(largest_contour)
            w_rot = min(rect[1][0], rect[1][1])
            h_rot = max(rect[1][0], rect[1][1])
            object_width = int(w_rot)
            object_height = int(h_rot)

            x_bb, y_bb, _, _ = cv2.boundingRect(largest_contour)
            if pixels_per_metric is not None:
                cv2.putText(display, f"W: {int(w_rot)}px ({w_rot / pixels_per_metric:.2f}mm)", (x_bb, y_bb - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"H: {int(h_rot)}px ({h_rot / pixels_per_metric:.2f}mm)", (x_bb, y_bb - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, f"W: {int(w_rot)}px", (x_bb, y_bb - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"H: {int(h_rot)}px", (x_bb, y_bb - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        if gauge_row is not None and tool_result is None:
            cv2.line(display, (0, gauge_row), (display.shape[1], gauge_row), (0, 200, 255), 1)
        if setting_gauge:
            cv2.putText(display, "CLICK ON THE GAUGE LINE", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)

        cv2.putText(display, "Y Accept | N Reject | M Measure tool | G Gauge line | C Calibrate | R Live",
                    (20, display.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)

        if show_manual_menu:
            cv2.rectangle(display, (50, 80), (550, 260), (40, 40, 40), -1)
            cv2.rectangle(display, (50, 80), (550, 260), (0, 255, 255), 2)
            cv2.putText(display, "MANUAL MEASUREMENT", (80, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "Press 1 : Contour Mode", (80, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Press 2 : Multi-Point Mode", (80, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ----- Manual Contour Mode -----
        if manual_mode == "CONTOUR":
            cv2.putText(display, "CONTOUR MODE - Click To Lock", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            for contour in contours:
                if cv2.pointPolygonTest(contour, (mouse_x, mouse_y), False) >= 0:
                    if not contour_locked:
                        selected_contour = contour
                    break

            if selected_contour is not None:
                color = (0, 255, 0) if contour_locked else (0, 255, 255)
                thickness = 3 if contour_locked else 2
                cv2.drawContours(display, [selected_contour], -1, color, thickness)

                rect = cv2.minAreaRect(selected_contour)
                box = np.int32(cv2.boxPoints(rect))
                cv2.drawContours(display, [box], 0, (0, 200, 255), 1)

                w_rot = min(rect[1][0], rect[1][1])
                h_rot = max(rect[1][0], rect[1][1])
                object_width = int(w_rot)
                object_height = int(h_rot)

                x_bb, y_bb, _, _ = cv2.boundingRect(selected_contour)
                if pixels_per_metric is not None:
                    cv2.putText(display, f"Width: {int(w_rot)}px ({w_rot / pixels_per_metric:.2f} mm)", (x_bb, y_bb - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(display, f"Height: {int(h_rot)}px ({h_rot / pixels_per_metric:.2f} mm)", (x_bb, y_bb - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(display, f"Width: {int(w_rot)}px", (x_bb, y_bb - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(display, f"Height: {int(h_rot)}px", (x_bb, y_bb - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if contour_locked:
                    cv2.putText(display, "OBJECT LOCKED", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    if not measurement_printed:
                        print("\nOBJECT LOCKED")
                        print("Width =", int(w_rot), "pixels")
                        print("Height =", int(h_rot), "pixels")
                        if pixels_per_metric is not None:
                            print(f"Width = {w_rot / pixels_per_metric:.2f} mm")
                            print(f"Height = {h_rot / pixels_per_metric:.2f} mm")
                        measurement_printed = True

        # ----- Multi-Point Mode -----
        elif manual_mode == "POINT":
            cv2.putText(display, f"MULTI-POINT MODE | SNAP: {'ON' if snap_enabled else 'OFF'} | K=Snap | Z=Undo",
                        (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if snap_enabled and snap_preview is not None:
                x, y = snap_preview
                cv2.rectangle(display, (x - 6, y - 6), (x + 6, y + 6), (0, 255, 255), 2)

            total_dist = 0.0
            for i, pt in enumerate(points):
                cv2.circle(display, pt, 5, (0, 0, 255), -1)
                cv2.putText(display, f"P{i + 1}", (pt[0] + 8, pt[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
                if i > 0:
                    prev_pt = points[i - 1]
                    cv2.line(display, prev_pt, pt, (255, 0, 0), 2)
                    seg_dist = float(np.hypot(pt[0] - prev_pt[0], pt[1] - prev_pt[1]))
                    total_dist += seg_dist
                    mid_x = (prev_pt[0] + pt[0]) // 2
                    mid_y = (prev_pt[1] + pt[1]) // 2
                    txt = f"{seg_dist:.1f}px" + (f" ({seg_dist / pixels_per_metric:.2f}mm)" if pixels_per_metric else "")
                    cv2.putText(display, txt, (mid_x + 5, mid_y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            if len(points) >= 2:
                cv2.putText(display, f"Segments: {len(points) - 1}", (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                txt = f"Total: {total_dist:.2f}px" + (f" ({total_dist / pixels_per_metric:.2f}mm)" if pixels_per_metric else "")
                cv2.putText(display, txt, (20, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw calibration status banner
    if pixels_per_metric is not None:
        cv2.putText(display, f"Calibrated: {pixels_per_metric:.4f} px/mm | D = Clear Calib",
                    (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    else:
        cv2.putText(display, "Not Calibrated (showing px) | C = Calibrate",
                    (20, display.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Vision Measurement System", display)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('s'):
        captured_frame = frame.copy()
        state = "CAPTURED"
        show_manual_menu = False
        manual_mode = None
        contour_locked = False
        selected_contour = None
        measurement_printed = False
        tool_result = None
        points = []
        print("Frame Captured")

    elif key == ord('m') and state == "CAPTURED":
        if pixels_per_metric is None:
            print("\nCalibrate first (C) - tool measurement needs pixels per mm.")
        else:
            gy = gauge_row if gauge_row is not None else captured_frame.shape[0] // 2
            try:
                tool_result = measure_tool(captured_frame, pixels_per_metric, gy, band_mm=tuple(args.band),
                                           tip=args.tip, z_camera_mm=args.z_camera)
                point = ("flat end" if tool_result.point_angle_deg is None
                         else f"point {tool_result.point_angle_deg:.1f} deg")
                print(f"\nTOOL T{args.tool}: length {tool_result.length_mm:.3f} mm | "
                      f"diameter {tool_result.diameter_mm:.3f} mm | {point}")
                for line in fanuc_g10(args.tool, tool_result.length_mm, tool_result.diameter_mm):
                    print("  " + line)
            except ValueError as e:
                tool_result = None
                print("\nTool measurement failed:", e)

    elif key == ord('g') and state == "CAPTURED":
        setting_gauge = True
        print("\nClick on the gauge line (holder reference face).")

    elif key == ord('y') and state == "CAPTURED":
        print("\nObject Accepted")
        if manual_mode == "POINT" and len(points) >= 2:
            total = 0.0
            for i in range(1, len(points)):
                seg = float(np.hypot(points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]))
                total += seg
                if pixels_per_metric is not None:
                    print(f"  Segment {i}: P{i}->P{i + 1} = {seg:.2f}px ({seg / pixels_per_metric:.2f}mm)")
                else:
                    print(f"  Segment {i}: P{i}->P{i + 1} = {seg:.2f}px")
            if pixels_per_metric is not None:
                print(f"  Total Path = {total:.2f}px ({total / pixels_per_metric:.2f}mm)")
            else:
                print(f"  Total Path = {total:.2f}px")
        else:
            print("Width =", object_width, "pixels")
            print("Height =", object_height, "pixels")
            if pixels_per_metric is not None:
                print(f"Width = {object_width / pixels_per_metric:.2f} mm")
                print(f"Height = {object_height / pixels_per_metric:.2f} mm")

    elif key == ord('n') and state == "CAPTURED":
        show_manual_menu = True
        tool_result = None
        print("\nObject Rejected")
        print("1 = Contour Mode")
        print("2 = Multi-Point Mode")

    elif key == ord('1') and show_manual_menu:
        manual_mode = "CONTOUR"
        show_manual_menu = False
        contour_locked = False
        selected_contour = None
        measurement_printed = False
        print("\nContour Mode Selected")

    elif key == ord('2') and show_manual_menu:
        manual_mode = "POINT"
        show_manual_menu = False
        points = []
        print("\nMulti-Point Mode Selected (Click to add points, Z to undo)")

    elif key == ord('k') and manual_mode == "POINT":
        snap_enabled = not snap_enabled
        print(f"Snap {'ON' if snap_enabled else 'OFF'}")

    elif key == ord('z') and manual_mode == "POINT":
        if len(points) > 0:
            removed = points.pop()
            print(f"Undo: Removed point P{len(points) + 1} at {removed}")

    elif key == ord('c'):
        if state == "LIVE":
            captured_frame = frame.copy()
        state = "CALIBRATING"
        calibration_point1 = None
        calibration_point2 = None
        run_calibration_prompt = False
        print("\nCalibration Mode. Click 2 points on a known reference object.")

    elif key == ord('d'):
        pixels_per_metric = None
        if os.path.exists(CONFIG_PATH):
            try:
                os.remove(CONFIG_PATH)
                print("\nCalibration cleared and config file deleted.")
            except Exception as e:
                print("Error deleting calibration file:", e)
        else:
            print("\nNo saved calibration found to clear.")

    elif key == ord('r'):
        state = "LIVE"
        show_manual_menu = False
        manual_mode = None
        contour_locked = False
        selected_contour = None
        measurement_printed = False
        tool_result = None
        setting_gauge = False
        calibration_point1 = None
        calibration_point2 = None
        points = []
        print("Live Mode")

    elif key == ord('t'):
        tracking_active = not tracking_active
        accumulated_dx = 0.0
        accumulated_dy = 0.0
        tracking_prev_gray = None
        tracking_prev_pts = None
        print(f"\nCamera tracking {'Started' if tracking_active else 'Stopped'}")

    elif key == ord('q'):
        break

if cap is not None:
    cap.release()
cv2.destroyAllWindows()
