from __future__ import annotations

import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
ULTRALYTICS_ROOT = REPO_ROOT / "ultralytics-8.3.163"
if str(ULTRALYTICS_ROOT) not in sys.path:
    sys.path.insert(0, str(ULTRALYTICS_ROOT))

from pedestrian_system.privacy.face_blur import FaceBlur


def main():
    cfg = {
        "enabled": True,
        # prefer gaussian for visible effect; can set 'mosaic'
        "blur_type": "gaussian",
        "blur_kernel": 51,
    }

    fb = FaceBlur(cfg)

    video_path = ULTRALYTICS_ROOT / "pedestrian_system" / "videos" / "test.mp4"
    out_dir = ULTRALYTICS_ROOT / "pedestrian_system" / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_img = out_dir / "test_face_blur_output.jpg"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"[Test] Cannot open video: {video_path}")
        return 1

    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        print("[Test] Failed to read frame from video")
        return 1

    out = fb.apply(frame)
    cv2.imwrite(str(out_img), out)
    print(f"[Test] Saved blurred frame to: {out_img}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
