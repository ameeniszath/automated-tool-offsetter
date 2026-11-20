import cv2
import numpy as np
import json
import os
import tkinter as tk
from tkinter import simpledialog, messagebox

# Initialize tkinter root and hide it
root = tk.Tk()
root.withdraw()

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "calibration_config.json")
pixels_per_metric = None

def load_calibration():
    global pixels_per_metric
    if os.path.exists(CONFIG_PATH):
        try:
            with open(CONFIG_PATH, 'r') as f:
                data = json.load(f)
                pixels_per_metric = data.get("pixels_per_metric", None)
                if pixels_per_metric is not None:
                    print(f"Loaded calibration: {pixels_per_metric:.4f} pixels/mm")
        except Exception as e:
            print("Error loading calibration:", e)

def save_calibration(val):
    global pixels_per_metric
    pixels_per_metric = val
    try:
        with open(CONFIG_PATH, 'w') as f:
            json.dump({"pixels_per_metric": pixels_per_metric}, f)
        print(f"Saved calibration: {pixels_per_metric:.4f} pixels/mm")
    except Exception as e:
        print("Error saving calibration:", e)

load_calibration()

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

def mouse_event(event, x, y, flags, param):
    global mouse_x, mouse_y, contour_locked, manual_mode
    global points, calibration_point1, calibration_point2, run_calibration_prompt, state, snap_enabled, snap_preview

    mouse_x = x
    mouse_y = y

    if event == cv2.EVENT_LBUTTONDOWN:
        if state == "CALIBRATING":
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

cap = cv2.VideoCapture(1)

cv2.namedWindow("Vision Measurement System")
cv2.setMouseCallback("Vision Measurement System", mouse_event)

while True:

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
                    dx_med = np.median(dxs)
                    dy_med = np.median(dys)
                    accumulated_dx += dx_med
                    accumulated_dy += dy_med
                    tracking_prev_pts = good_next.reshape(-1, 1, 2)
                else:
                    tracking_prev_pts = None
            else:
                tracking_prev_pts = None
                
        if tracking_prev_pts is None or len(tracking_prev_pts) < 15:
            pts = cv2.goodFeaturesToTrack(
                gray, maxCorners=100, qualityLevel=0.01, minDistance=10
            )
            if pts is not None:
                tracking_prev_pts = pts
            else:
                tracking_prev_pts = None
        tracking_prev_gray = gray.copy()
    else:
        tracking_prev_gray = None
        tracking_prev_pts = None

    # Check if we need to run the calibration prompt in the main thread
    if run_calibration_prompt:
        run_calibration_prompt = False

        dx = calibration_point2[0] - calibration_point1[0]
        dy = calibration_point2[1] - calibration_point1[1]
        pixel_dist = (dx**2 + dy**2) ** 0.5
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

        # Center Reference Crosshair
        h, w = display.shape[:2]
        cx = w // 2
        cy = h // 2

        cv2.line(display, (cx - 25, cy), (cx + 25, cy), (0, 255, 0), 1)
        cv2.line(display, (cx, cy - 25), (cx, cy + 25), (0, 255, 0), 1)
        cv2.circle(display, (cx, cy), 3, (0, 255, 0), -1)

        # Draw tracked features and stats if tracking is active
        if tracking_active and tracking_prev_pts is not None:
            for pt in tracking_prev_pts:
                px, py = pt[0]
                cv2.circle(display, (int(px), int(py)), 3, (255, 255, 0), -1)

            dist_px = (accumulated_dx**2 + accumulated_dy**2)**0.5
            if pixels_per_metric is not None:
                dx_mm = accumulated_dx / pixels_per_metric
                dy_mm = accumulated_dy / pixels_per_metric
                dist_mm = dist_px / pixels_per_metric
                track_txt = f"Cam Shift: DX={dx_mm:.2f}mm, DY={dy_mm:.2f}mm, Dist={dist_mm:.2f}mm"
            else:
                track_txt = f"Cam Shift: DX={accumulated_dx:.1f}px, DY={accumulated_dy:.1f}px, Dist={dist_px:.1f}px"

            cv2.putText(
                display,
                track_txt,
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )
            cv2.putText(
                display,
                "TRACKING ACTIVE (Press T to Reset/Stop)",
                (20, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1
            )

        cv2.putText(
            display,
            "Press S to Capture | C to Calibrate | T to Track",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )
    elif state == "CALIBRATING":
        display = captured_frame.copy()
        cv2.putText(
            display,
            "CALIBRATING - CLICK 2 POINTS ON REFERENCE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
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
        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        # Snap preview calculation
        if manual_mode == "POINT":
            snap_preview = None
            if snap_enabled and len(contours)>0:
                contour=max(contours,key=cv2.contourArea)
                best_dist=1e9
                best_pt=None
                for p in contour:
                    px,py=p[0]
                    d=((px-mouse_x)**2+(py-mouse_y)**2)**0.5
                    if d<best_dist:
                        best_dist=d
                        best_pt=(int(px),int(py))
                if best_pt is not None and best_dist<=SNAP_RADIUS:
                    snap_preview=best_pt


        # ----- Circle Detection (HoughCircles) -----
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.2,
            minDist=50,
            param1=100,
            param2=40,
            minRadius=10,
            maxRadius=0
        )


        # ----- Auto Contour Mode (Actual Outline + Rotated Bounding Box) -----
        if len(contours) > 0 and manual_mode is None:
            largest_contour = max(contours, key=cv2.contourArea)

            # Draw actual contour outline (green)
            cv2.drawContours(display, [largest_contour], -1, (0, 255, 0), 2)

            # Calculate width and height from rotated rect (used for measurements only)
            rect = cv2.minAreaRect(largest_contour)
            w_rot = min(rect[1][0], rect[1][1])
            h_rot = max(rect[1][0], rect[1][1])
            object_width = int(w_rot)
            object_height = int(h_rot)

            # Position measurement text
            x_bb, y_bb, _, _ = cv2.boundingRect(largest_contour)
            if pixels_per_metric is not None:
                w_mm = w_rot / pixels_per_metric
                h_mm = h_rot / pixels_per_metric
                cv2.putText(display, f"W: {int(w_rot)}px ({w_mm:.2f}mm)", (x_bb, y_bb - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"H: {int(h_rot)}px ({h_mm:.2f}mm)", (x_bb, y_bb - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            else:
                cv2.putText(display, f"W: {int(w_rot)}px", (x_bb, y_bb - 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(display, f"H: {int(h_rot)}px", (x_bb, y_bb - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.putText(
            display,
            "Y = Accept | N = Reject | C = Calibrate | R = Live",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            2
        )

        if show_manual_menu:
            cv2.rectangle(display, (50, 80), (550, 260), (40, 40, 40), -1)
            cv2.rectangle(display, (50, 80), (550, 260), (0, 255, 255), 2)
            cv2.putText(display, "MANUAL MEASUREMENT", (80, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
            cv2.putText(display, "Press 1 : Contour Mode", (80, 170),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display, "Press 2 : Multi-Point Mode", (80, 220),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        # ----- Manual Contour Mode -----
        if manual_mode == "CONTOUR":
            cv2.putText(
                display,
                "CONTOUR MODE - Click To Lock",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            for contour in contours:
                inside = cv2.pointPolygonTest(
                    contour,
                    (mouse_x, mouse_y),
                    False
                )
                if inside >= 0:
                    if not contour_locked:
                        selected_contour = contour
                    break

            if selected_contour is not None:
                # Draw actual contour outline
                color = (0, 255, 0) if contour_locked else (0, 255, 255)
                thickness = 3 if contour_locked else 2
                cv2.drawContours(display, [selected_contour], -1, color, thickness)

                # Draw rotated rect overlay
                rect = cv2.minAreaRect(selected_contour)
                box = cv2.boxPoints(rect)
                box = np.int32(box)
                cv2.drawContours(display, [box], 0, (0, 200, 255), 1)

                w_rot = min(rect[1][0], rect[1][1])
                h_rot = max(rect[1][0], rect[1][1])
                object_width = int(w_rot)
                object_height = int(h_rot)

                x_bb, y_bb, _, _ = cv2.boundingRect(selected_contour)
                if pixels_per_metric is not None:
                    w_mm = w_rot / pixels_per_metric
                    h_mm = h_rot / pixels_per_metric
                    cv2.putText(display, f"Width: {int(w_rot)}px ({w_mm:.2f} mm)", (x_bb, y_bb - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(display, f"Height: {int(h_rot)}px ({h_mm:.2f} mm)", (x_bb, y_bb - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    cv2.putText(display, f"Width: {int(w_rot)}px", (x_bb, y_bb - 40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    cv2.putText(display, f"Height: {int(h_rot)}px", (x_bb, y_bb - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if contour_locked:
                    cv2.putText(
                        display,
                        "OBJECT LOCKED",
                        (20, 120),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                    if not measurement_printed:
                        print("")
                        print("OBJECT LOCKED")
                        print("Width =", int(w_rot), "pixels")
                        print("Height =", int(h_rot), "pixels")
                        if pixels_per_metric is not None:
                            print(f"Width = {w_rot / pixels_per_metric:.2f} mm")
                            print(f"Height = {h_rot / pixels_per_metric:.2f} mm")
                        measurement_printed = True

        # ----- Multi-Point Mode -----
        elif manual_mode == "POINT":
            cv2.putText(
                display,
                f"MULTI-POINT MODE | SNAP: {'ON' if snap_enabled else 'OFF'} | K=Snap | Z=Undo",
                (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            if snap_enabled and snap_preview is not None:
                x,y=snap_preview
                cv2.rectangle(display,(x-6,y-6),(x+6,y+6),(0,255,255),2)

            total_dist = 0.0

            # Draw points and segments
            for i, pt in enumerate(points):
                cv2.circle(display, pt, 5, (0, 0, 255), -1)
                cv2.putText(display, f"P{i+1}", (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

                if i > 0:
                    prev_pt = points[i - 1]
                    cv2.line(display, prev_pt, pt, (255, 0, 0), 2)

                    dx = abs(pt[0] - prev_pt[0])
                    dy = abs(pt[1] - prev_pt[1])
                    seg_dist = (dx**2 + dy**2) ** 0.5
                    total_dist += seg_dist

                    # Label at midpoint of each segment
                    mid_x = (prev_pt[0] + pt[0]) // 2
                    mid_y = (prev_pt[1] + pt[1]) // 2
                    if pixels_per_metric is not None:
                        seg_mm = seg_dist / pixels_per_metric
                        cv2.putText(display, f"{seg_dist:.1f}px ({seg_mm:.2f}mm)",
                                    (mid_x + 5, mid_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)
                    else:
                        cv2.putText(display, f"{seg_dist:.1f}px",
                                    (mid_x + 5, mid_y - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 0), 1)

            # Display summary
            if len(points) >= 2:
                y_offset = 120
                cv2.putText(display, f"Segments: {len(points) - 1}", (20, y_offset),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                y_offset += 30
                if pixels_per_metric is not None:
                    total_mm = total_dist / pixels_per_metric
                    cv2.putText(display, f"Total: {total_dist:.2f}px ({total_mm:.2f}mm)", (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                else:
                    cv2.putText(display, f"Total: {total_dist:.2f}px", (20, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Draw calibration status banner
    if pixels_per_metric is not None:
        cv2.putText(
            display,
            f"Calibrated: {pixels_per_metric:.4f} px/mm | D = Clear Calib",
            (20, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )
    else:
        cv2.putText(
            display,
            "Not Calibrated (showing px) | C = Calibrate",
            (20, display.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )

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
        points = []
        print("Frame Captured")

    elif key == ord('y') and state == "CAPTURED":
        print("")
        print("Object Accepted")
        if manual_mode == "POINT" and len(points) >= 2:
            total = 0.0
            for i in range(1, len(points)):
                dx = abs(points[i][0] - points[i-1][0])
                dy = abs(points[i][1] - points[i-1][1])
                seg = (dx**2 + dy**2) ** 0.5
                total += seg
                if pixels_per_metric is not None:
                    print(f"  Segment {i}: P{i}->P{i+1} = {seg:.2f}px ({seg/pixels_per_metric:.2f}mm)")
                else:
                    print(f"  Segment {i}: P{i}->P{i+1} = {seg:.2f}px")
            if pixels_per_metric is not None:
                print(f"  Total Path = {total:.2f}px ({total/pixels_per_metric:.2f}mm)")
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
        print("")
        print("Object Rejected")
        print("1 = Contour Mode")
        print("2 = Multi-Point Mode")

    elif key == ord('1') and show_manual_menu:
        manual_mode = "CONTOUR"
        show_manual_menu = False
        contour_locked = False
        selected_contour = None
        measurement_printed = False
        print("")
        print("Contour Mode Selected")

    elif key == ord('2') and show_manual_menu:
        manual_mode = "POINT"
        show_manual_menu = False
        points = []
        print("")
        print("Multi-Point Mode Selected (Click to add points, Z to undo)")

    elif key == ord('k') and manual_mode == "POINT":
        snap_enabled = not snap_enabled
        print(f"Snap {'ON' if snap_enabled else 'OFF'}")

    elif key == ord('z') and manual_mode == "POINT":
        if len(points) > 0:
            removed = points.pop()
            print(f"Undo: Removed point P{len(points)+1} at {removed}")

    elif key == ord('c'):
        if state == "LIVE":
            captured_frame = frame.copy()
        state = "CALIBRATING"
        calibration_point1 = None
        calibration_point2 = None
        run_calibration_prompt = False

        print("")
        print("Calibration Mode. Click 2 points on a known reference object.")

    elif key == ord('d'):
        pixels_per_metric = None
        if os.path.exists(CONFIG_PATH):
            try:
                os.remove(CONFIG_PATH)
                print("")
                print("Calibration cleared and config file deleted.")
            except Exception as e:
                print("Error deleting calibration file:", e)
        else:
            print("")
            print("No saved calibration found to clear.")

    elif key == ord('r'):
        state = "LIVE"
        show_manual_menu = False
        manual_mode = None
        contour_locked = False
        selected_contour = None
        measurement_printed = False
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
        print("")
        print(f"Camera tracking {'Started' if tracking_active else 'Stopped'}")

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
