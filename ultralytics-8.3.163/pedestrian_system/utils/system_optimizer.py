from __future__ import annotations

import os
import sys
from pathlib import Path

try:
    import torch
except Exception:
    torch = None


def apply_system_optimizations():
    results = {}

    results["cpu_affinity"] = _apply_cpu_affinity()
    results["gpu_memory"] = _configure_gpu_memory()
    results["thread_priority"] = _set_thread_priority()
    results["opencv_backend"] = _configure_opencv_backend()
    results["torch_threads"] = _configure_torch_threads()
    results["mkl"] = _configure_mkl()

    return results


def _apply_cpu_affinity() -> dict:
    try:
        if sys.platform == "win32":
            proc_handle = _get_current_process_handle()
            if proc_handle is not None:
                import ctypes
                import ctypes.wintypes
                kernel32 = ctypes.windll.kernel32
                num_cores = os.cpu_count() or 4
                mask = (1 << num_cores) - 1
                kernel32.SetProcessAffinityMask(proc_handle, mask)
                return {"status": "ok", "cores": num_cores, "mask": hex(mask)}
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}

    try:
        affinity = os.sched_getaffinity(0) if hasattr(os, "sched_getaffinity") else None
        if affinity is not None:
            return {"status": "ok", "cores": len(affinity)}
    except Exception:
        pass

    return {"status": "skipped"}


def _get_current_process_handle():
    try:
        import ctypes
        import ctypes.wintypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        return ctypes.wintypes.HANDLE(handle)
    except Exception:
        return None


def _configure_gpu_memory() -> dict:
    if torch is None or not torch.cuda.is_available():
        return {"status": "skipped", "reason": "CUDA not available"}

    results = {}
    try:
        alloc_conf = os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "")
        if "expandable_segments:True" not in alloc_conf:
            new_conf = alloc_conf + ",expandable_segments:True" if alloc_conf else "expandable_segments:True"
            os.environ["PYTORCH_CUDA_ALLOC_CONF"] = new_conf
            results["expandable_segments"] = True

        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            if hasattr(torch, "set_float32_matmul_precision"):
                torch.set_float32_matmul_precision("high")
            results["tf32_enabled"] = True

        torch.backends.cudnn.benchmark = True
        results["cudnn_benchmark"] = True

        device_name = torch.cuda.get_device_name(0) or "Unknown GPU"
        total_mem = torch.cuda.get_device_properties(0).total_mem
        results["device"] = device_name
        results["total_memory_gb"] = round(total_mem / (1024**3), 1)
    except Exception as exc:
        results["status"] = "error"
        results["detail"] = str(exc)

    results["status"] = "ok"
    return results


def _set_thread_priority() -> dict:
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.GetCurrentThread()
            THREAD_PRIORITY_HIGHEST = 2
            if kernel32.SetThreadPriority(handle, THREAD_PRIORITY_HIGHEST):
                return {"status": "ok", "level": "HIGHEST"}
        else:
            os.nice(-10)
            return {"status": "ok", "level": "nice=-10"}
    except Exception as exc:
        return {"status": "skipped", "detail": str(exc)}


def _configure_opencv_backend() -> dict:
    import cv2
    results = {}
    try:
        backend_status = {}
        for name, flag in [
            ("D3D11", cv2.CAP_MSMF),
            ("MSMF", cv2.CAP_MSMF),
            ("FFMPEG", cv2.CAP_FFMPEG),
        ]:
            try:
                backend_status[name] = "available"
            except Exception:
                backend_status[name] = "unavailable"
        results["available_backends"] = backend_status
    except Exception:
        pass

    try:
        cv2.setNumThreads(min(4, os.cpu_count() or 4))
        results["cv2_threads"] = cv2.getNumThreads()
    except Exception:
        pass

    try:
        if hasattr(cv2, "cuda"):
            results["opencv_cuda"] = cv2.cuda.getCudaEnabledDeviceCount() > 0
    except Exception:
        results["opencv_cuda"] = False

    results["status"] = "ok"
    return results


def _configure_torch_threads() -> dict:
    if torch is None:
        return {"status": "skipped"}

    try:
        import multiprocessing
        num_phys = multiprocessing.cpu_count() or 4
        torch.set_num_threads(min(num_phys, 8))
        torch.set_num_interop_threads(min(num_phys // 2, 4))
        return {
            "status": "ok",
            "num_threads": torch.get_num_threads(),
            "num_interop": torch.get_num_interop_threads() if hasattr(torch, "get_num_interop_threads") else "N/A",
        }
    except Exception as exc:
        return {"status": "error", "detail": str(exc)}


def _configure_mkl() -> dict:
    try:
        if "MKL_NUM_THREADS" not in os.environ:
            os.environ["MKL_NUM_THREADS"] = str(min(4, os.cpu_count() or 4))
        if "OMP_NUM_THREADS" not in os.environ:
            os.environ["OMP_NUM_THREADS"] = str(min(4, os.cpu_count() or 4))
        return {"status": "ok", "MKL_NUM_THREADS": os.environ.get("MKL_NUM_THREADS")}
    except Exception:
        return {"status": "skipped"}


def get_system_info_report() -> dict:
    info = {
        "platform": sys.platform,
        "cpu_count": os.cpu_count(),
    }

    try:
        import psutil
        info["cpu_percent"] = psutil.cpu_percent(interval=0.1)
        info["memory_percent"] = psutil.virtual_memory().percent
        info["memory_total_gb"] = round(psutil.virtual_memory().total / (1024**3), 1)
    except Exception:
        pass

    if torch is not None and torch.cuda.is_available():
        try:
            info["cuda_available"] = True
            info["cuda_device"] = torch.cuda.get_device_name(0)
            info["cuda_memory_gb"] = round(torch.cuda.get_device_properties(0).total_mem / (1024**3), 1)
            info["cuda_compute_capability"] = f"{torch.cuda.get_device_properties(0).major}.{torch.cuda.get_device_properties(0).minor}"
        except Exception:
            pass
    else:
        info["cuda_available"] = False

    return info
