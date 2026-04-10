from __future__ import annotations

import os
# Windows + Torch 稳定性设置
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import torch
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.backends.mkldnn.enabled = False
except ImportError:
    pass

import argparse
import csv
import time
from collections import defaultdict, deque
from pathlib import Path
import traceback

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from counting.line_counter import LineCounter
from counting.zone_counter import ZoneCounterManager
from detector.yolo_detector import YOLODetector
from tracker.deepsort_wrapper import DeepSortTracker
from utils.config import load_config
from utils.filters import filter_tracks_by_roi, StaticTrackFilter
from utils.io_utils import EventLogger, ensure_dir, save_summary_json
from utils.visualization import (
    draw_counters_panel,
    draw_line_counter,
    draw_track_history,
    draw_tracks,
    draw_zone_counters,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/pedestrian_demo.yaml")
    return parser.parse_args()


def parse_source(source_value):
    if isinstance(source_value, int):
        return source_value
    if isinstance(source_value, str) and source_value.isdigit():
        return int(source_value)
    return source_value


def save_flow_csv(flow_stats: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    max_minute = max(flow_stats.keys(), default=-1)
    with open(save_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.writer(f)
        writer.writerow(["minute", "up", "down", "total"])
        for minute_idx in range(max_minute + 1):
            up = int(flow_stats[minute_idx]["up"])
            down = int(flow_stats[minute_idx]["down"])
            writer.writerow([minute_idx, up, down, up + down])


def save_flow_plot(flow_stats: dict, save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    max_minute = max(flow_stats.keys(), default=-1)
    minutes = list(range(max_minute + 1)) if max_minute >= 0 else [0]
    up_values = [int(flow_stats[m]["up"]) for m in minutes]
    down_values = [int(flow_stats[m]["down"]) for m in minutes]
    total_values = [u + d for u, d in zip(up_values, down_values)]

    plt.figure(figsize=(10, 5))
    plt.plot(minutes, up_values, marker="o", label="Up")
    plt.plot(minutes, down_values, marker="o", label="Down")
    plt.plot(minutes, total_values, marker="o", label="Total")
    plt.xlabel("Minute")
    plt.ylabel("People Count")
    plt.title("Pedestrian Flow Trend")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def main():
    print(">>> program start")

    try:
        args = parse_args()
        cfg = load_config(args.config)

        source = parse_source(cfg["source"])
        print("SOURCE:", source)
        if isinstance(source, int):
            print("VIDEO EXISTS: camera source")
        else:
            print("VIDEO EXISTS:", Path(source).exists())

        model_path = cfg["detector"]["model_path"]
        print("MODEL:", model_path)
        print("MODEL EXISTS:", Path(model_path).exists())

        output_dir = Path(cfg["output_dir"])
        ensure_dir(output_dir)

        cap = cv2.VideoCapture(source)
        print("cap opened:", cap.isOpened())
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {source}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 1e-6:
            fps = float(cfg["output"].get("fallback_fps", 25.0))

        raw_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        raw_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        proc_width, proc_height = 1280, 800
        print(f"Video size: {raw_width}x{raw_height}, fps={fps}")
        print(f"Process size: {proc_width}x{proc_height}")

        save_video = bool(cfg["output"].get("save_video", True))
        display = bool(cfg["output"].get("display", True))

        video_path = output_dir / cfg["output"].get("video_name", "result.mp4")
        events_csv_path = output_dir / cfg["output"].get("events_csv_name", "events.csv")
        summary_json_path = output_dir / cfg["output"].get("summary_json_name", "summary.json")

        # 新增：趋势图输出文件
        flow_csv_path = output_dir / cfg["output"].get("flow_csv_name", "flow.csv")
        flow_plot_path = output_dir / cfg["output"].get("flow_plot_name", "flow_trend.png")

        writer = None
        if save_video:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(video_path), fourcc, fps, (proc_width, proc_height))
            if not writer.isOpened():
                raise RuntimeError(f"Failed to create output video: {video_path}")

        try:
            detector = YOLODetector(cfg["detector"])
        except Exception:
            print("⚠️ YOLO GPU load failed, fallback to CPU")
            cfg["detector"]["device"] = "cpu"
            detector = YOLODetector(cfg["detector"])

        tracker = DeepSortTracker(cfg["tracker"])

        counting_cfg = cfg.get("counting", {})
        roi_cfg = counting_cfg.get("roi", {})

        static_filter_cfg = cfg.get("static_filter", {})
        static_filter_enabled = bool(static_filter_cfg.get("enabled", False))
        static_filter = StaticTrackFilter(static_filter_cfg)
        hide_static_boxes = bool(static_filter_cfg.get("hide_static_boxes", True))

        line_counter = None
        if counting_cfg.get("line", {}).get("enabled", False):
            line_counter = LineCounter(counting_cfg["line"])

        zone_manager = ZoneCounterManager(counting_cfg.get("zones", []))
        event_logger = EventLogger(events_csv_path)

        visualization_cfg = cfg.get("visualization", {})
        trail_length = int(visualization_cfg.get("trail_length", 30))
        show_confidence = bool(visualization_cfg.get("show_confidence", True))
        box_thickness = int(visualization_cfg.get("box_thickness", 2))
        draw_roi = bool(visualization_cfg.get("draw_roi", True))

        debug_cfg = cfg.get("debug", {})
        draw_count_points = bool(debug_cfg.get("draw_count_points", True))
        log_every_n_frames = int(debug_cfg.get("log_every_n_frames", 300))
        crossing_ttl = int(debug_cfg.get("crossing_ttl", 25))
        show_minute_stats = bool(debug_cfg.get("show_minute_stats", True))

        debug_cross_buffer = []
        track_history = defaultdict(lambda: deque(maxlen=trail_length))

        # 新增：每分钟客流统计
        flow_stats = defaultdict(lambda: {"up": 0, "down": 0})

        frame_idx = 0
        total_time = 0.0
        total_frames = 0

        print(">>> start processing...")

        while True:
            success, frame = cap.read()
            if not success:
                print(">>> video end")
                break

            start = time.time()
            frame = cv2.resize(frame, (proc_width, proc_height))

            # ===== detection =====
            detections = detector.detect(frame)
            all_tracks = tracker.update(frame, detections)
            track_map = {tr.track_id: tr for tr in all_tracks}

            # ===== ROI =====
            roi_tracks = all_tracks
            if roi_cfg.get("enabled", False):
                roi_tracks = filter_tracks_by_roi(
                    all_tracks,
                    polygon=roi_cfg.get("polygon", []),
                    anchor_point=roi_cfg.get("anchor_point", "bottom_center"),
                )

            # ===== history =====
            for tr in roi_tracks:
                p = tr.count_point("bottom_center")
                track_history[tr.track_id].append((int(p[0]), int(p[1])))

            # ===== static filter =====
            static_ids = set()
            if static_filter_enabled:
                static_ids = static_filter.update(roi_tracks, frame_idx)

            counting_tracks = [tr for tr in roi_tracks if tr.track_id not in static_ids]

            if log_every_n_frames > 0 and frame_idx % log_every_n_frames == 0:
                print(
                    f"[frame {frame_idx}] all={len(all_tracks)} roi={len(roi_tracks)} counting={len(counting_tracks)}"
                )

            # ===== line counter =====
            line_events = []
            if line_counter is not None:
                line_events = line_counter.update(counting_tracks, frame_idx)
                for event in line_events:
                    event_logger.add_event(event)
                    
                    debug_cross_buffer.append({
                        "event": event,
                        "ttl": crossing_ttl,
                    })

                    # 趋势图统计：按分钟累计 up/down
                    minute_idx = int((frame_idx / fps) // 60)

                    direction = event.get("direction")
                    if direction is None:
                        direction = event.get("label")

                    if direction == "up":
                        flow_stats[minute_idx]["up"] += 1
                    elif direction == "down":
                        flow_stats[minute_idx]["down"] += 1

            zone_events = zone_manager.update(counting_tracks, frame_idx)
            for event in zone_events:
                event_logger.add_event(event)

            # ===== visualization =====
            annotated = frame.copy()

            if roi_cfg.get("enabled", False) and draw_roi:
                pts = np.array(roi_cfg.get("polygon", []), dtype=np.int32)
                if len(pts) >= 3:
                    cv2.polylines(annotated, [pts], True, (255, 0, 255), 2)
                    label_pos = (int(pts[0][0]) + 8, int(pts[0][1]) - 8)
                    cv2.putText(
                        annotated,
                        "counting_roi",
                        label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 0, 255),
                        2,
                    )

            if draw_count_points:
                for tr in counting_tracks:
                    pt = tr.count_point("bottom_center")
                    x, y = int(pt[0]), int(pt[1])

                    cv2.circle(annotated, (x, y), 5, (0, 255, 0), -1)
                    cv2.putText(
                        annotated,
                        f"{tr.track_id}",
                        (x, y - 8),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (0, 255, 0),
                        2,
                    )

            tracks_for_draw = counting_tracks if hide_static_boxes else roi_tracks
            annotated = draw_tracks(
                annotated,
                tracks_for_draw,
                show_conf=show_confidence,
                box_thickness=box_thickness,
            )

            visible_ids = {tr.track_id for tr in tracks_for_draw}
            visible_history = {
                tid: hist for tid, hist in track_history.items() if tid in visible_ids
            }
            annotated = draw_track_history(annotated, visible_history)

            if line_counter is not None:
                annotated = draw_line_counter(annotated, line_counter)

            annotated = draw_zone_counters(annotated, zone_manager)
            annotated = draw_counters_panel(annotated, line_counter, zone_manager)
            
            new_buffer = []
            for item in debug_cross_buffer:
                e = item["event"]
                ttl = item["ttl"]
                point = e.get("point")
                if point is None:
                    track_id = e.get("track_id")
                    tr = track_map.get(track_id)
                    if tr is not None:
                        try:
                            point = tr.count_point("bottom_center")
                        except Exception:
                            point = None
                if point is None:
                    continue
                x, y = int(point[0]), int(point[1])
                direction = (e.get("direction") or e.get("label") or "").lower()
                side = e.get("side", "")
                
                cv2.circle(annotated, (x, y), 12, (0, 0, 255), -1)
                
                if direction == "up":
                    text = "Up"
                elif direction == "down":
                    text = "Down"
                else:
                    text = "Cross"
                    
                cv2.putText(
                    annotated,
                    text,
                    (x + 12, y - 12),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2,
                )
                
                if side != "":
                    cv2.putText(
                        annotated,
                        f"side:{side}",
                        (x + 12, y + 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                    )
                ttl -= 1
                if ttl > 0:
                    new_buffer.append({
                        "event": e,
                        "ttl": ttl,
                    })
            debug_cross_buffer = new_buffer

            # 可选：把当前 minute 的趋势统计显示在画面左下角
            if show_minute_stats:
                current_minute = int((frame_idx / fps) // 60)
                up_now = flow_stats[current_minute]["up"]
                down_now = flow_stats[current_minute]["down"]
                cv2.putText(
                    annotated,
                    f"Minute {current_minute}: up={up_now} down={down_now}",
                    (20, proc_height - 20),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            # ===== display =====
            if display:
                cv2.imshow("Pedestrian Flow System", annotated)
                if cv2.waitKey(1) & 0xFF in [27, ord("q")]:
                    break

            if writer is not None:
                writer.write(annotated)

            # 清理旧轨迹历史
            active_ids = {tr.track_id for tr in all_tracks}
            for tid in list(track_history.keys()):
                if tid not in active_ids and tid not in visible_ids:
                    track_history.pop(tid, None)

            frame_idx += 1
            total_frames += 1
            total_time += time.time() - start

        cap.release()
        if writer is not None:
            writer.release()
        cv2.destroyAllWindows()

        avg_fps = total_frames / total_time if total_time > 0 else 0.0

        # 新增：输出趋势数据
        save_flow_csv(flow_stats, flow_csv_path)
        save_flow_plot(flow_stats, flow_plot_path)

        summary = {
            "source": str(source),
            "frames_processed": total_frames,
            "avg_pipeline_fps": round(avg_fps, 3),
            "detector": detector.summary(),
            "tracker": tracker.summary(),
            "roi_enabled": bool(roi_cfg.get("enabled", False)),
            "static_filter_enabled": static_filter_enabled,
            "line_counter": line_counter.summary() if line_counter is not None else None,
            "zone_counter": zone_manager.summary(),
            "output_video": str(video_path) if save_video else None,
            "events_csv": str(events_csv_path),
            "flow_csv": str(flow_csv_path),
            "flow_plot": str(flow_plot_path),
        }

        event_logger.flush()
        save_summary_json(summary_json_path, summary)

        print("=" * 70)
        print("Processing finished.")
        print(f"Frames processed: {total_frames}")
        print(f"Average pipeline FPS: {avg_fps:.3f}")
        print(f"Events CSV: {events_csv_path}")
        print(f"Flow CSV: {flow_csv_path}")
        print(f"Flow Plot: {flow_plot_path}")
        print(f"Summary JSON: {summary_json_path}")
        if save_video:
            print(f"Output video: {video_path}")
        print("=" * 70)

    except Exception:
        print("\n❌ PROGRAM CRASHED")
        traceback.print_exc()

if __name__ == "__main__":
    import sys
    if "--gui" in sys.argv:
        from gui import run_app
        run_app()
    else:
        main()
