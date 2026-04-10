import cv2
from pathlib import Path

image_path = Path(r"E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163\pedestrian_system\tools\test_frame.jpg")
points = []

def mouse_callback(event, x, y, flags, param):
    global img, points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        print(f"Point {len(points)}: ({x}, {y})")

        cv2.circle(img, (x, y), 6, (0, 0, 255), -1)
        cv2.putText(
            img,
            f"({x},{y})",
            (x + 10, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2
        )
        cv2.imshow("pick_points", img)

img = cv2.imread(str(image_path))
if img is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

cv2.imshow("pick_points", img)
cv2.setMouseCallback("pick_points", mouse_callback)

print("Left click to pick points.")
print("Press q to quit.")
print("All clicked points will be printed in the terminal.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
print("Final points:", points)
