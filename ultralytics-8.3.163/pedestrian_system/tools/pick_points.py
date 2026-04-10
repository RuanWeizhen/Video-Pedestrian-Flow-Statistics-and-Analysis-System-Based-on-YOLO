import cv2
from pathlib import Path

image_path = Path(r"E:\Video Pedestrian Flow Statistics and Analysis System Based on YOLO\ultralytics-8.3.163\pedestrian_system\tools\test_frame.jpg")

# 最大显示尺寸，可按你的屏幕调整
MAX_DISPLAY_WIDTH = 1400
MAX_DISPLAY_HEIGHT = 900

points = []

img_original = cv2.imread(str(image_path))
if img_original is None:
    raise FileNotFoundError(f"Cannot load image: {image_path}")

orig_h, orig_w = img_original.shape[:2]
print(f"Original image size: width={orig_w}, height={orig_h}")

scale = min(MAX_DISPLAY_WIDTH / orig_w, MAX_DISPLAY_HEIGHT / orig_h, 1.0)
display_w = int(orig_w * scale)
display_h = int(orig_h * scale)

img_display = cv2.resize(img_original, (display_w, display_h), interpolation=cv2.INTER_AREA)
img_show = img_display.copy()

print(f"Display image size: width={display_w}, height={display_h}")
print(f"Scale: {scale}")

def mouse_callback(event, x, y, flags, param):
    global img_show, points

    if event == cv2.EVENT_LBUTTONDOWN:
        # 显示图坐标 -> 原图坐标
        orig_x = int(round(x / scale))
        orig_y = int(round(y / scale))

        points.append((orig_x, orig_y))
        print(f"Point {len(points)}: display=({x}, {y}), original=({orig_x}, {orig_y})")

        # 在显示图上画点击点
        cv2.circle(img_show, (x, y), 5, (0, 0, 255), -1)
        cv2.putText(
            img_show,
            f"({orig_x},{orig_y})",
            (x + 8, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2
        )
        cv2.imshow("pick_points", img_show)

cv2.namedWindow("pick_points", cv2.WINDOW_NORMAL)
cv2.imshow("pick_points", img_show)
cv2.setMouseCallback("pick_points", mouse_callback)

print("Left click to pick points.")
print("Press q to quit.")
print("Printed coordinates are ORIGINAL image coordinates.")

while True:
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break

cv2.destroyAllWindows()
print("Final points:", points)
