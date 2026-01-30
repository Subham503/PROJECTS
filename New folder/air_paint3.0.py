import cv2
import numpy as np

# --- Camera setup ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# --- Canvas setup ---
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# --- Default brush settings ---
brush_color = (0, 0, 255)
brush_size = 8

# --- Kernel for noise reduction ---
kernel = np.ones((5, 5), np.uint8)

# --- History stacks ---
undo_history = []
redo_history = []

# --- Variables ---
x_prev, y_prev = 0, 0
drawing = False  # Track whether a stroke is active

# --- Color Palette Setup ---
palette = [
    {"color": (0, 0, 255), "name": "Red"},
    {"color": (0, 255, 0), "name": "Green"},
    {"color": (255, 0, 0), "name": "Blue"},
    {"color": (0, 255, 255), "name": "Yellow"},
    {"color": (255, 0, 255), "name": "Pink"},
    {"color": (0, 0, 0), "name": "Eraser"},
]
palette_y = 650  # vertical position of palette

# --- Helper Functions ---
def smooth_point(prev, new, alpha=0.3):
    if prev == 0:
        return new
    return int(prev * (1 - alpha) + new * alpha)


def store_history(img):
    small = cv2.resize(img, (640, 360))
    undo_history.append(small)


def restore_history(history_list):
    if history_list:
        small = history_list.pop()
        return cv2.resize(small, (1280, 720))
    return None


# --- HSV Trackbar setup ---
def nothing(x):
    pass


cv2.namedWindow("HSV Control", cv2.WINDOW_NORMAL)
cv2.resizeWindow("HSV Control", 400, 300)
cv2.createTrackbar("LH", "HSV Control", 100, 179, nothing)
cv2.createTrackbar("LS", "HSV Control", 150, 255, nothing)
cv2.createTrackbar("LV", "HSV Control", 0, 255, nothing)
cv2.createTrackbar("UH", "HSV Control", 140, 179, nothing)
cv2.createTrackbar("US", "HSV Control", 255, 255, nothing)
cv2.createTrackbar("UV", "HSV Control", 255, 255, nothing)

print("🎨 Air Painter Pro 3.0 started!")
print("Use a blue object to draw.")
print("Now you can pick colors by hovering over palette boxes at the bottom of the screen!")

# --- Main Loop ---
while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # --- Get HSV values from trackbars ---
    lh = cv2.getTrackbarPos("LH", "HSV Control")
    ls = cv2.getTrackbarPos("LS", "HSV Control")
    lv = cv2.getTrackbarPos("LV", "HSV Control")
    uh = cv2.getTrackbarPos("UH", "HSV Control")
    us = cv2.getTrackbarPos("US", "HSV Control")
    uv = cv2.getTrackbarPos("UV", "HSV Control")

    lower_color = np.array([lh, ls, lv])
    upper_color = np.array([uh, us, uv])

    # --- Mask for detecting color ---
    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

    # --- Find contours ---
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(c)

        if 1500 < area < 25000:
            x, y, w, h = cv2.boundingRect(c)
            cx, cy = x + w // 2, y + h // 2
            cx = smooth_point(x_prev, cx)
            cy = smooth_point(y_prev, cy)

            cv2.circle(frame, (cx, cy), brush_size, brush_color, -1)

            # --- Palette Interaction ---
            for i, p in enumerate(palette):
                x1 = 50 + i * 200
                y1 = palette_y
                x2 = x1 + 150
                y2 = y1 + 50
                if x1 < cx < x2 and y1 < cy < y2:
                    brush_color = p["color"]
                    print(f"🎨 Brush changed to {p['name']}")
                    x_prev, y_prev = 0, 0
                    drawing = False
                    break

            if cy < palette_y:  # Only draw above palette
                if not drawing:
                    store_history(canvas.copy())
                    redo_history.clear()
                    drawing = True

                if x_prev == 0 and y_prev == 0:
                    x_prev, y_prev = cx, cy
                else:
                    cv2.line(canvas, (x_prev, y_prev), (cx, cy), brush_color, brush_size)
                    x_prev, y_prev = cx, cy
        else:
            drawing = False
            x_prev, y_prev = 0, 0
    else:
        if drawing:
            drawing = False
        x_prev, y_prev = 0, 0

    # --- Combine frame and canvas ---
    combined = cv2.addWeighted(frame, 0.7, canvas, 0.5, 0)

    # --- Semi-transparent palette overlay ---
    overlay = combined.copy()
    for i, p in enumerate(palette):
        x1 = 50 + i * 200
        y1 = palette_y
        x2 = x1 + 150
        y2 = y1 + 50
        cv2.rectangle(overlay, (x1, y1), (x2, y2), p["color"], -1)
    cv2.addWeighted(overlay, 0.6, combined, 0.4, 0, combined)

    # --- Palette borders and labels ---
    for i, p in enumerate(palette):
        x1 = 50 + i * 200
        y1 = palette_y
        x2 = x1 + 150
        y2 = y1 + 50
        cv2.rectangle(combined, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv2.putText(combined, p["name"], (x1 + 15, y1 + 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    # --- Info text (top-left) ---
    mode_text = "Eraser" if brush_color == (0, 0, 0) else "Brush"
    cv2.putText(combined, f'{mode_text} | Size: {brush_size}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(combined, f'HSV: [{lh},{ls},{lv}] - [{uh},{us},{uv}]', (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 180), 2)

    # --- Controls text (top-right corner) ---
    instructions = 'R G B Y P = Color | + - = Size | E = Eraser | C = Clear | S = Save | U = Undo | O = Redo | Q = Quit'
    (text_width, _), _ = cv2.getTextSize(instructions, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.putText(combined, instructions, (1280 - text_width - 20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 255, 180), 2)

    # --- Display windows ---
    cv2.imshow("🎨 Air Painter Pro 3.0", combined)
   # cv2.imshow("Mask", mask)

    # --- Key Controls ---
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[:] = 0
        undo_history.clear()
        redo_history.clear()
        print("🧹 Canvas cleared!")
    elif key == ord('s'):
        cv2.imwrite('My_Air_Paint_Art.png', canvas)
        print("✅ Drawing saved as My_Air_Paint_Art.png")
    elif key == ord('u'):
        restored = restore_history(undo_history)
        if restored is not None:
            redo_history.append(canvas.copy())
            canvas[:] = restored
            print("↩ Undo successful!")
        else:
            print("⚠ Nothing to undo.")
    elif key == ord('o'):
        restored = restore_history(redo_history)
        if restored is not None:
            undo_history.append(cv2.resize(canvas.copy(), (640, 360)))
            canvas[:] = restored
            print("🔁 Redo successful!")
        else:
            print("⚠ Nothing to redo.")
    elif key == ord('+') or key == ord('='):
        brush_size = min(brush_size + 2, 40)
        print(f"Brush size increased to {brush_size}")
    elif key == ord('-') or key == ord('_'):
        brush_size = max(2, brush_size - 2)
        print(f"Brush size decreased to {brush_size}")

cap.release()
cv2.destroyAllWindows()
