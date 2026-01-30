import cv2
import numpy as np
import math

# --- Camera setup ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# --- Canvas setup ---
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# --- Brush settings ---
brush_color = (0, 0, 255)
brush_size = 8
mode = 'free'
drawing = False
shape_start = None
shape_preview = None
x_prev, y_prev = 0, 0

# --- Undo/Redo ---
undo_history = []
redo_history = []

# --- Palette setup ---
palette = [
    {"color": (0, 0, 255), "name": "Red"},
    {"color": (0, 255, 0), "name": "Green"},
    {"color": (255, 0, 0), "name": "Blue"},
    {"color": (0, 255, 255), "name": "Yellow"},
    {"color": (255, 0, 255), "name": "Pink"},
    {"color": (255, 255, 255), "name": "White"},
    {"color": (0, 0, 0), "name": "Eraser"}
]
palette_y = 650

# --- HSV range for blue pointer detection ---
lower_blue = np.array([100, 120, 70])
upper_blue = np.array([140, 255, 255])

kernel = np.ones((5, 5), np.uint8)

print("🎨 Air Painter Pro 5.1 — Pointer Edition with Live Shapes + On-screen Instructions")

def store_history(img):
    undo_history.append(cv2.resize(img.copy(), (640, 360)))
    if len(undo_history) > 15:
        undo_history.pop(0)

def restore_history(history_list):
    if history_list:
        small = history_list.pop()
        return cv2.resize(small, (1280, 720))
    return None

def smooth_point(prev, new, alpha=0.3):
    if prev == 0:
        return new
    return int(prev * (1 - alpha) + new * alpha)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cx, cy = None, None

    display_canvas = canvas.copy()

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area > 1500:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = smooth_point(x_prev, x + w // 2), smooth_point(y_prev, y + h // 2)
            cv2.circle(frame, (cx, cy), 10, (0, 255, 255), -1)

            # --- Palette selection detection ---
            for i, p in enumerate(palette):
                x1 = 50 + i * 180
                y1 = palette_y
                x2 = x1 + 150
                y2 = y1 + 50
                if x1 < cx < x2 and y1 < cy < y2:
                    brush_color = p["color"]
                    print(f"🎨 Color changed to {p['name']}")
                    x_prev, y_prev = 0, 0
                    shape_start = None
                    drawing = False
                    break

            # --- Drawing area ---
            if cy < palette_y - 20:
                if mode == 'free':
                    if x_prev == 0 and y_prev == 0:
                        x_prev, y_prev = cx, cy
                    cv2.line(canvas, (x_prev, y_prev), (cx, cy), brush_color, brush_size, cv2.LINE_AA)
                    x_prev, y_prev = cx, cy

                elif mode in ['line', 'rect', 'circle']:
                    if not drawing:
                        shape_start = (cx, cy)
                        drawing = True
                    else:
                        shape_preview = (cx, cy)
                        # --- Live shape preview ---
                        if mode == 'line':
                            cv2.line(display_canvas, shape_start, shape_preview, brush_color, brush_size)
                        elif mode == 'rect':
                            cv2.rectangle(display_canvas, shape_start, shape_preview, brush_color, brush_size)
                        elif mode == 'circle':
                            radius = int(math.hypot(shape_preview[0] - shape_start[0], shape_preview[1] - shape_start[1]))
                            cv2.circle(display_canvas, shape_start, radius, brush_color, brush_size)
        else:
            # --- When pointer leaves screen ---
            if drawing and shape_start and shape_preview:
                store_history(canvas)
                if mode == 'line':
                    cv2.line(canvas, shape_start, shape_preview, brush_color, brush_size)
                elif mode == 'rect':
                    cv2.rectangle(canvas, shape_start, shape_preview, brush_color, brush_size)
                elif mode == 'circle':
                    radius = int(math.hypot(shape_preview[0] - shape_start[0], shape_preview[1] - shape_start[1]))
                    cv2.circle(canvas, shape_start, radius, brush_color, brush_size)
            drawing = False
            shape_start = None
            shape_preview = None
            x_prev, y_prev = 0, 0
    else:
        # --- No pointer detected ---
        if drawing and shape_start and shape_preview:
            store_history(canvas)
            if mode == 'line':
                cv2.line(canvas, shape_start, shape_preview, brush_color, brush_size)
            elif mode == 'rect':
                cv2.rectangle(canvas, shape_start, shape_preview, brush_color, brush_size)
            elif mode == 'circle':
                radius = int(math.hypot(shape_preview[0] - shape_start[0], shape_preview[1] - shape_start[1]))
                cv2.circle(canvas, shape_start, radius, brush_color, brush_size)
        drawing = False
        shape_start = None
        shape_preview = None
        x_prev, y_prev = 0, 0

    # --- Combine the frame and canvas ---
    combined = cv2.addWeighted(frame, 0.4, display_canvas, 1.0, 0)

    # --- Draw color palette ---
    for i, p in enumerate(palette):
        x1 = 50 + i * 180
        y1 = palette_y
        x2 = x1 + 150
        y2 = y1 + 50
        cv2.rectangle(combined, (x1, y1), (x2, y2), p["color"], -1)
        cv2.rectangle(combined, (x1, y1), (x2, y2), (255, 255, 255), 2)
        text_color = (0, 0, 0) if p["color"] == (255, 255, 255) else (255, 255, 255)
        cv2.putText(combined, p["name"], (x1 + 10, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

    # --- Top-left info (mode and brush size) ---
    cv2.putText(combined, f"Mode: {mode.upper()} | Size: {brush_size}",
                (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    # --- Top-right corner instructions ---
    cv2.putText(combined, "f=Free  l=Line  r=Rect  c=Circle", (750, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    cv2.putText(combined, "u/o=Undo/Redo  s=Save  x=Clear", (750, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
    cv2.putText(combined, "+/-=Brush Size  q=Quit", (750, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)

    cv2.imshow("🎨 Air Painter Pro 5.1", combined)

    # --- Keyboard controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('f'):
        mode = 'free'; print("✏ Mode: Free Draw")
    elif key == ord('l'):
        mode = 'line'; print("📏 Mode: Line")
    elif key == ord('r'):
        mode = 'rect'; print("⬛ Mode: Rectangle")
    elif key == ord('c'):
        mode = 'circle'; print("⚪ Mode: Circle")
    elif key == ord('+') or key == ord('='):
        brush_size = min(brush_size + 2, 40); print(f"🖌 Brush: {brush_size}")
    elif key == ord('-') or key == ord('_'):
        brush_size = max(2, brush_size - 2); print(f"🖌 Brush: {brush_size}")
    elif key == ord('u'):
        restored = restore_history(undo_history)
        if restored is not None:
            redo_history.append(canvas.copy())
            canvas[:] = restored
            print("↩ Undo")
    elif key == ord('o'):
        restored = restore_history(redo_history)
        if restored is not None:
            undo_history.append(cv2.resize(canvas.copy(), (640, 360)))
            canvas[:] = restored
            print("↪ Redo")
    elif key == ord('x'):
        store_history(canvas)
        canvas[:] = 0
        print("🧹 Canvas Cleared")
    elif key == ord('s'):
        cv2.imwrite("AirPainter_Pro5_Result.png", canvas)
        print("💾 Saved as AirPainter_Pro5_Result.png")

cap.release()
cv2.destroyAllWindows()
