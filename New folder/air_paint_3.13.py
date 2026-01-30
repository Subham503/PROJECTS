import cv2
import numpy as np

# --- Camera setup ---
cap = cv2.VideoCapture(0)
cap.set(3, 1280)
cap.set(4, 720)

# --- Canvas setup ---
canvas = np.zeros((720, 1280, 3), dtype=np.uint8)

# --- Brush settings ---
brush_color = (0, 0, 255)  # Red
brush_size = 8

# --- Choose the color of the object to track (change if needed) ---
# Example: Blue object
lower_color = np.array([100, 150, 50])
upper_color = np.array([140, 255, 255])

x_prev, y_prev = 0, 0
kernel = np.ones((5, 5), np.uint8)

def smooth_point(prev, new, alpha=0.3):
    if prev == 0:
        return new
    return int(prev * (1 - alpha) + new * alpha)

while True:
    success, frame = cap.read()
    if not success:
        break

    frame = cv2.flip(frame, 1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    mask = cv2.inRange(hsv, lower_color, upper_color)
    mask = cv2.erode(mask, kernel, iterations=1)
    mask = cv2.dilate(mask, kernel, iterations=2)
    mask = cv2.GaussianBlur(mask, (7, 7), 0)

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

            if x_prev == 0 and y_prev == 0:
                x_prev, y_prev = cx, cy
            else:
                cv2.line(canvas, (x_prev, y_prev), (cx, cy), brush_color, brush_size)
                x_prev, y_prev = cx, cy
        else:
            x_prev, y_prev = 0, 0
    else:
        x_prev, y_prev = 0, 0

    combined = cv2.addWeighted(frame, 0.7, canvas, 0.5, 0)
    cv2.putText(combined, "Press Q to quit | C clear | S save | R,G,B,Y,P change color | E eraser | +/- brush size",
                (10, 700), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    cv2.imshow("🎨 Air Paint (Python 3.13 Compatible)", combined)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        canvas[:] = 0
        print("🧹 Canvas cleared!")
    elif key == ord('s'):
        cv2.imwrite('My_Air_Paint_Art.png', canvas)
        print("✅ Saved as My_Air_Paint_Art.png")
    elif key == ord('r'):
        brush_color = (0, 0, 255)
    elif key == ord('g'):
        brush_color = (0, 255, 0)
    elif key == ord('b'):
        brush_color = (255, 0, 0)
    elif key == ord('y'):
        brush_color = (0, 255, 255)
    elif key == ord('p'):
        brush_color = (255, 0, 255)
    elif key == ord('e'):
        brush_color = (0, 0, 0)
    elif key == ord('+') or key == ord('='):
        brush_size = min(brush_size + 2, 40)
        print(f"Brush size increased to {brush_size}")
    elif key == ord('-') or key == ord('_'):
        brush_size = max(2, brush_size - 2)
        print(f"Brush size decreased to {brush_size}")

cap.release()
cv2.destroyAllWindows()
