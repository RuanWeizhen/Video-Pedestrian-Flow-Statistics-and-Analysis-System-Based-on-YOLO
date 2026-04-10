import cv2

video_path = "videos/test.mp4"
frame_id = 100   # 你想截第几帧

cap = cv2.VideoCapture(video_path)
cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
ret, frame = cap.read()

if not ret:
    raise RuntimeError("Failed to read frame.")

cv2.imwrite("test_frame.jpg", frame)
print("Saved test_frame.jpg")
cap.release()
