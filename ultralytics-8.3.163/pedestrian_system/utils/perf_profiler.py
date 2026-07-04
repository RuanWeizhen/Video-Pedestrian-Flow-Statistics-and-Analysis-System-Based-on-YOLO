from __future__ import annotations

import json
import os
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict

try:
    import psutil
except Exception:
    psutil = None

try:
    import torch
except Exception:
    torch = None


class PerformanceProfiler:
    FRAME_WINDOW = 120
    GPU_SAMPLE_INTERVAL = 1.0

    def __init__(self):
        self._fps_history = deque(maxlen=self.FRAME_WINDOW)
        self._preprocess_history = deque(maxlen=self.FRAME_WINDOW)
        self._inference_history = deque(maxlen=self.FRAME_WINDOW)
        self._postprocess_history = deque(maxlen=self.FRAME_WINDOW)
        self._gpu_util_history = deque(maxlen=self.FRAME_WINDOW)
        self._cpu_util_history = deque(maxlen=self.FRAME_WINDOW)
        self._memory_history = deque(maxlen=self.FRAME_WINDOW)
        self._last_gpu_sample = 0.0
        self._frame_count = 0
        self._session_start = time.perf_counter()
        self._baseline_snapshot: dict | None = None

    def record_frame(self, preprocess_ms: float, inference_ms: float, postprocess_ms: float,
                     total_ms: float):
        self._frame_count += 1
        self._preprocess_history.append(preprocess_ms)
        self._inference_history.append(inference_ms)
        self._postprocess_history.append(postprocess_ms)

        fps = (1000.0 / total_ms) if total_ms > 1e-6 else 0.0
        self._fps_history.append(fps)

        now = time.perf_counter()
        if now - self._last_gpu_sample >= self.GPU_SAMPLE_INTERVAL:
            self._last_gpu_sample = now
            self._sample_system()

    def _sample_system(self):
        if psutil is not None:
            try:
                self._cpu_util_history.append(psutil.cpu_percent(interval=None))
                self._memory_history.append(psutil.virtual_memory().percent)
            except Exception:
                pass

        if torch is not None and torch.cuda.is_available():
            try:
                util = _read_nvidia_gpu_util()
                self._gpu_util_history.append(float(util) if util is not None else 0.0)
            except Exception:
                pass

    def snapshot(self) -> dict:
        elapsed = max(time.perf_counter() - self._session_start, 0.001)
        avg_fps = (self._frame_count / elapsed) if elapsed > 0 else 0.0
        return {
            "fps": round(avg_fps, 2),
            "frames": self._frame_count,
            "elapsed_sec": round(elapsed, 1),
            "preprocess_ms": round(_avg(self._preprocess_history), 2),
            "inference_ms": round(_avg(self._inference_history), 2),
            "postprocess_ms": round(_avg(self._postprocess_history), 2),
            "gpu_util_pct": round(_avg(self._gpu_util_history), 1),
            "cpu_util_pct": round(_avg(self._cpu_util_history), 1),
            "memory_pct": round(_avg(self._memory_history), 1),
        }

    def baseline(self) -> dict:
        if self._baseline_snapshot is None:
            self._baseline_snapshot = self.snapshot()
        return self._baseline_snapshot

    def delta(self) -> dict:
        base = self.baseline()
        now = self.snapshot()
        return {k: round(now[k] - base.get(k, 0), 2) for k in now}

    def export_report(self, output_path: str | Path, label: str = "性能报告") -> str:
        now_data = self.snapshot()
        base_data = self.baseline()
        report = {
            "title": label,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "current": now_data,
            "baseline": base_data,
            "delta": {k: round(now_data.get(k, 0) - base_data.get(k, 0), 2)
                      for k in now_data},
            "recommendations": self._generate_recommendations(now_data),
        }
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        return str(out)

    def _generate_recommendations(self, data: dict) -> list[str]:
        tips = []
        prep = data.get("preprocess_ms", 0)
        inf = data.get("inference_ms", 0)
        post = data.get("postprocess_ms", 0)
        if prep > inf * 0.3:
            tips.append("预处理耗时占比偏高，建议启用 GPU resize/normalize 或异步帧读取")
        if post > inf * 0.5:
            tips.append("后处理耗时过高，建议使用 GPU NMS 或减少可视化复杂度")
        if data.get("gpu_util_pct", 0) < 30:
            tips.append("GPU 利用率偏低，考虑增大 batch size 或启用 FP16")
        if data.get("cpu_util_pct", 0) > 80:
            tips.append("CPU 占用过高，检查是否有非推理计算瓶颈")
        return tips

    def reset(self):
        self._fps_history.clear()
        self._preprocess_history.clear()
        self._inference_history.clear()
        self._postprocess_history.clear()
        self._gpu_util_history.clear()
        self._cpu_util_history.clear()
        self._memory_history.clear()
        self._frame_count = 0
        self._session_start = time.perf_counter()
        self._baseline_snapshot = None


def _avg(deq: deque) -> float:
    if not deq:
        return 0.0
    return sum(deq) / len(deq)


def _read_nvidia_gpu_util() -> float | None:
    try:
        result = os.popen(
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits"
        ).read().strip()
        return float(result.split("\n")[0].strip())
    except Exception:
        return None
