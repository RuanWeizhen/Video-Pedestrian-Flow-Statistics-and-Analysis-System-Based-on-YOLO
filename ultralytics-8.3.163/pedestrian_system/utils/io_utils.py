from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def make_event(frame_idx: int, event_type: str, target: str, track_id: int, value: str) -> Dict:
    return {
        "frame_idx": frame_idx,
        "event_type": event_type,
        "target": target,
        "track_id": track_id,
        "value": value,
    }


class EventLogger:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.events: List[Dict] = []

    def add_event(self, event: Dict) -> None:
        self.events.append(event)

    def _write_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=["frame_idx", "event_type", "target", "track_id", "value"],
            )
            writer.writeheader()

            clean_events = []
            for e in self.events:
                clean_events.append({
                    "frame_idx": e.get("frame_idx"),
                    "event_type": e.get("event_type"),
                    "target": e.get("target"),
                    "track_id": e.get("track_id"),
                    "value": e.get("value"),
                })

            writer.writerows(clean_events)

    def flush(self) -> None:
        try:
            self._write_csv(self.csv_path)
        except PermissionError:
            # 原文件被 Excel / WPS / 编辑器占用时，自动换名保存
            base = self.csv_path.stem
            suffix = self.csv_path.suffix
            parent = self.csv_path.parent

            fallback_path = None
            for i in range(1, 1000):
                candidate = parent / f"{base}_new_{i}{suffix}"
                if not candidate.exists():
                    fallback_path = candidate
                    break

            if fallback_path is None:
                raise RuntimeError("Failed to find an available fallback CSV filename.")

            self._write_csv(fallback_path)
            print(f"⚠️ {self.csv_path} 被占用，已改为保存到: {fallback_path}")


def save_summary_json(json_path: Path, summary: Dict) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
