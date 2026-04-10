from __future__ import annotations

import argparse
import copy
from pathlib import Path

import cv2
import yaml



def parse_source(source_value):
    if isinstance(source_value, int):
        return source_value
    if isinstance(source_value, str) and source_value.isdigit():
        return int(source_value)
    return source_value


def ensure_counting_cfg(cfg: dict) -> None:
    cfg.setdefault("counting", {})
    cfg["counting"].setdefault("roi", {})
    cfg["counting"].setdefault("line", {})
    cfg["counting"].setdefault("zones", [])


def load_yaml(config_path: Path) -> dict:
    with config_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_yaml(config_path: Path, cfg: dict) -> None:
    backup_path = config_path.with_suffix(config_path.suffix + ".bak")
    if not backup_path.exists():
        backup_path.write_text(
            config_path.read_text(encoding="utf-8"), encoding="utf-8"
        )

    with config_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)


class ROIAndLineEditor:
    def __init__(
        self,
        frame,
        cfg: dict,
        config_path: Path,
        proc_width: int,
        proc_height: int,
        sync_zone: bool = True,
    ):
        self.original_frame = frame
        self.frame = frame.copy()
        self.cfg = cfg
        self.config_path = config_path
        self.proc_width = proc_width
        self.proc_height = proc_height
        self.sync_zone = sync_zone

        ensure_counting_cfg(self.cfg)

        roi_cfg = self.cfg["counting"].get("roi", {})
        line_cfg = self.cfg["counting"].get("line", {})

        self.roi_points = [tuple(map(int, p)) for p in roi_cfg.get("polygon", [])]
        self.line_points = [tuple(map(int, p)) for p in line_cfg.get("points", [])]

        self.mode = "roi"  # roi / line
        self.window_name = "ROI + Line Editor"

    def on_mouse(self, event, x, y, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return

        if self.mode == "roi":
            self.roi_points.append((int(x), int(y)))
        else:
            if len(self.line_points) >= 2:
                self.line_points = []
            self.line_points.append((int(x), int(y)))

    def draw(self):
        canvas = self.frame.copy()

        # ROI
        for i, p in enumerate(self.roi_points):
            cv2.circle(canvas, p, 4, (255, 0, 255), -1)
            cv2.putText(
                canvas,
                f"R{i}",
                (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 255),
                1,
            )

        if len(self.roi_points) >= 2:
            for i in range(len(self.roi_points) - 1):
                cv2.line(canvas, self.roi_points[i], self.roi_points[i + 1], (255, 0, 255), 2)

        if len(self.roi_points) >= 3:
            cv2.polylines(
                canvas,
                [self._to_np_points(self.roi_points)],
                True,
                (255, 0, 255),
                2,
            )

        # Line
        for i, p in enumerate(self.line_points):
            cv2.circle(canvas, p, 5, (0, 0, 255), -1)
            cv2.putText(
                canvas,
                f"L{i}",
                (p[0] + 6, p[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
            )

        if len(self.line_points) == 2:
            cv2.line(canvas, self.line_points[0], self.line_points[1], (0, 0, 255), 2)

        # HUD
        lines = [
            f"Mode: {self.mode.upper()}",
            "Mouse Left: add point",
            "1: ROI mode",
            "2: LINE mode",
            "U: undo current mode",
            "C: clear current mode",
            f"Z: zone sync {'ON' if self.sync_zone else 'OFF'}",
            "S: save to YAML",
            "Q / ESC: quit",
        ]

        """
        鼠标左键：添加点

        1：切换到 ROI 模式

        2：切换到 Line 模式

        U：撤销当前模式最后一个点

        C：清空当前模式所有点

        Z：切换是否把 zones[0].polygon 同步成 ROI

        S：保存到 YAML

        Q 或 ESC：退出
        """

        y0 = 28
        for i, txt in enumerate(lines):
            cv2.putText(
                canvas,
                txt,
                (16, y0 + i * 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 255),
                2,
            )

        return canvas

    def _to_np_points(self, points):
        import numpy as np
        return np.array(points, dtype=np.int32)

    def undo(self):
        if self.mode == "roi":
            if self.roi_points:
                self.roi_points.pop()
        else:
            if self.line_points:
                self.line_points.pop()

    def clear_current(self):
        if self.mode == "roi":
            self.roi_points = []
        else:
            self.line_points = []

    def save(self):
        if len(self.roi_points) < 3:
            raise ValueError("ROI 至少需要 3 个点。")
        if len(self.line_points) != 2:
            raise ValueError("Line 必须恰好 2 个点。")

        cfg = copy.deepcopy(self.cfg)
        ensure_counting_cfg(cfg)

        cfg["counting"]["roi"]["enabled"] = True
        cfg["counting"]["roi"]["anchor_point"] = cfg["counting"]["roi"].get(
            "anchor_point", "bottom_center"
        )
        cfg["counting"]["roi"]["polygon"] = [[int(x), int(y)] for x, y in self.roi_points]

        cfg["counting"]["line"]["enabled"] = True
        cfg["counting"]["line"]["anchor_point"] = cfg["counting"]["line"].get(
            "anchor_point", "bottom_center"
        )
        cfg["counting"]["line"]["points"] = [[int(x), int(y)] for x, y in self.line_points]

        if self.sync_zone:
            zones = cfg["counting"].get("zones", [])
            if len(zones) == 0:
                zones.append(
                    {
                        "name": "main_area",
                        "enabled": False,
                        "anchor_point": "bottom_center",
                        "stale_after_frames": 45,
                        "cooldown_frames": 15,
                        "dedup_window_frames": 12,
                        "dedup_distance_px": 50,
                        "polygon": [],
                    }
                )
                cfg["counting"]["zones"] = zones

            cfg["counting"]["zones"][0]["polygon"] = [
                [int(x), int(y)] for x, y in self.roi_points
            ]

        save_yaml(self.config_path, cfg)
        self.cfg = cfg
        print(f"✅ 已保存到: {self.config_path}")
        print("ROI:", cfg["counting"]["roi"]["polygon"])
        print("Line:", cfg["counting"]["line"]["points"])

    def run(self):
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.proc_width, self.proc_height)
        cv2.setMouseCallback(self.window_name, self.on_mouse)

        while True:
            canvas = self.draw()
            cv2.imshow(self.window_name, canvas)
            key = cv2.waitKey(20) & 0xFF

            if key in (27, ord("q")):
                break
            elif key == ord("1"):
                self.mode = "roi"
            elif key == ord("2"):
                self.mode = "line"
            elif key in (ord("u"), ord("U")):
                self.undo()
            elif key in (ord("c"), ord("C")):
                self.clear_current()
            elif key in (ord("z"), ord("Z")):
                self.sync_zone = not self.sync_zone
            elif key in (ord("s"), ord("S")):
                try:
                    self.save()
                except Exception as e:
                    print(f"❌ 保存失败: {e}")

        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="ROI / Line Editor for pedestrian_demo.yaml")
    parser.add_argument(
        "--config",
        type=str,
        default="config/pedestrian_demo.yaml",
        help="YAML config path",
    )
    parser.add_argument(
        "--proc-width",
        type=int,
        default=1280,
        help="Processing width (must match main.py resize width)",
    )
    parser.add_argument(
        "--proc-height",
        type=int,
        default=800,
        help="Processing height (must match main.py resize height)",
    )
    parser.add_argument(
        "--source",
        type=str,
        default=None,
        help="Optional override source",
    )
    args = parser.parse_args()

    config_path = Path(args.config)
    cfg = load_yaml(config_path)

    source = args.source if args.source is not None else cfg["source"]
    source = parse_source(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频源: {source}")

    ok, frame = cap.read()
    cap.release()

    if not ok or frame is None:
        raise RuntimeError("无法读取视频第一帧。")

    frame = cv2.resize(frame, (args.proc_width, args.proc_height))

    editor = ROIAndLineEditor(
        frame=frame,
        cfg=cfg,
        config_path=config_path,
        proc_width=args.proc_width,
        proc_height=args.proc_height,
        sync_zone=True,
    )
    editor.run()


if __name__ == "__main__":
    main()
