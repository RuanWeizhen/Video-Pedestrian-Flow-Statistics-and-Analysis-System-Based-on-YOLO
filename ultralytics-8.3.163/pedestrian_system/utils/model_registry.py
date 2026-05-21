from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from .paths import writable_path


def _now_text() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _normalize_path(model_path: str) -> str:
    try:
        return str(Path(model_path).expanduser().resolve())
    except Exception:
        return str(model_path or "").strip()


def registry_file_path() -> Path:
    return writable_path("outputs/model_registry.json")


def infer_framework(model_path: str) -> str:
    suffix = Path(model_path).suffix.lower()
    if suffix == ".onnx":
        return "onnx"
    if suffix in {".engine", ".trt"}:
        return "tensorrt"
    if suffix in {".pt", ".pth"}:
        return "pytorch"
    if suffix in {".torchscript", ".ts"}:
        return "torchscript"
    return suffix.lstrip(".") or "unknown"


def _make_model_record(
    model_path: str,
    name: str | None = None,
    version: str | None = None,
    notes: str = "",
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    device: str | int | None = None,
    framework: str | None = None,
) -> dict:
    now = _now_text()
    normalized_path = _normalize_path(model_path)
    model_name = name or Path(normalized_path).stem or "模型"
    return {
        "id": f"model_{uuid4().hex[:12]}",
        "name": str(model_name),
        "version": str(version or "v1.0"),
        "path": normalized_path,
        "framework": str(framework or infer_framework(normalized_path)),
        "status": "inactive",
        "notes": str(notes or ""),
        "conf": float(conf) if conf is not None else None,
        "iou": float(iou) if iou is not None else None,
        "imgsz": int(imgsz) if imgsz is not None else None,
        "device": str(device) if device is not None else None,
        "created_at": now,
        "updated_at": now,
        "evaluation_count": 0,
        "last_eval_at": "",
        "last_eval_type": "",
        "last_eval_dataset": "",
        "last_eval_metrics": {},
    }


def normalize_registry(registry: dict | None, fallback_model_path: str = "") -> dict:
    data = deepcopy(registry) if isinstance(registry, dict) else {}
    data.setdefault("active_model_id", "")
    data.setdefault("models", [])
    data.setdefault("evaluations", [])
    data.setdefault("created_at", _now_text())
    data.setdefault("updated_at", _now_text())

    models = []
    for item in data.get("models", []):
        if not isinstance(item, dict):
            continue
        path = _normalize_path(item.get("path", ""))
        if not path:
            continue
        record = _make_model_record(
            model_path=path,
            name=item.get("name"),
            version=item.get("version"),
            notes=item.get("notes", ""),
            conf=item.get("conf"),
            iou=item.get("iou"),
            imgsz=item.get("imgsz"),
            device=item.get("device"),
            framework=item.get("framework"),
        )
        for key in (
            "id",
            "status",
            "created_at",
            "updated_at",
            "evaluation_count",
            "last_eval_at",
            "last_eval_type",
            "last_eval_dataset",
            "last_eval_metrics",
        ):
            if key in item and item.get(key) is not None:
                record[key] = deepcopy(item.get(key))
        record["id"] = str(item.get("id") or record["id"])
        record["status"] = str(item.get("status") or record["status"])
        record["evaluation_count"] = int(item.get("evaluation_count") or 0)
        record["last_eval_metrics"] = dict(item.get("last_eval_metrics") or {})
        models.append(record)

    data["models"] = models

    evaluations = []
    for item in data.get("evaluations", []):
        if isinstance(item, dict):
            evaluations.append(dict(item))
    data["evaluations"] = evaluations

    active_model_id = str(data.get("active_model_id") or "")
    if not active_model_id and models:
        active_model_id = str(models[0].get("id", ""))
    data["active_model_id"] = active_model_id

    if fallback_model_path:
        ensure_model_record(
            data,
            fallback_model_path,
            name=Path(fallback_model_path).stem,
            version="current",
            notes="当前配置模型",
            make_active=not bool(data.get("models")),
        )

    active = get_active_model(data)
    if active is not None:
        for item in data["models"]:
            item["status"] = "active" if str(item.get("id")) == str(active.get("id")) else "inactive"

    data["updated_at"] = _now_text()
    return data


def load_registry(default_model_path: str = "") -> dict:
    path = registry_file_path()
    if path.exists():
        try:
            with path.open("r", encoding="utf-8") as f:
                raw = json.load(f)
        except Exception:
            raw = {}
    else:
        raw = {}
    return normalize_registry(raw, fallback_model_path=default_model_path)


def save_registry(registry: dict) -> Path:
    path = registry_file_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    data = normalize_registry(registry)
    data["updated_at"] = _now_text()
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return path


def get_model_by_id(registry: dict, model_id: str) -> dict | None:
    model_id = str(model_id or "")
    if not model_id:
        return None
    for item in registry.get("models", []):
        if str(item.get("id", "")) == model_id:
            return item
    return None


def get_model_by_path(registry: dict, model_path: str) -> dict | None:
    normalized_path = _normalize_path(model_path)
    if not normalized_path:
        return None
    for item in registry.get("models", []):
        if _normalize_path(item.get("path", "")) == normalized_path:
            return item
    return None


def get_active_model(registry: dict) -> dict | None:
    active_id = str(registry.get("active_model_id") or "")
    if active_id:
        item = get_model_by_id(registry, active_id)
        if item is not None:
            return item
    models = registry.get("models", [])
    return models[0] if models else None


def ensure_model_record(
    registry: dict,
    model_path: str,
    name: str | None = None,
    version: str | None = None,
    notes: str = "",
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
    device: str | int | None = None,
    framework: str | None = None,
    make_active: bool = False,
) -> dict:
    normalized_path = _normalize_path(model_path)
    if not normalized_path:
        raise ValueError("model_path is required")

    existing = get_model_by_path(registry, normalized_path)
    if existing is not None:
        if name:
            existing["name"] = str(name)
        if version:
            existing["version"] = str(version)
        if notes:
            existing["notes"] = str(notes)
        if conf is not None:
            existing["conf"] = float(conf)
        if iou is not None:
            existing["iou"] = float(iou)
        if imgsz is not None:
            existing["imgsz"] = int(imgsz)
        if device is not None:
            existing["device"] = str(device)
        if framework:
            existing["framework"] = str(framework)
        existing["updated_at"] = _now_text()
        if make_active:
            registry["active_model_id"] = str(existing.get("id"))
        return existing

    record = _make_model_record(
        normalized_path,
        name=name,
        version=version,
        notes=notes,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        framework=framework,
    )
    registry.setdefault("models", []).append(record)
    if make_active or not registry.get("active_model_id"):
        registry["active_model_id"] = record["id"]
    registry["updated_at"] = _now_text()
    return record


def activate_model(registry: dict, model_id: str) -> dict:
    model = get_model_by_id(registry, model_id)
    if model is None:
        raise KeyError(f"Model not found: {model_id}")
    registry["active_model_id"] = str(model.get("id"))
    for item in registry.get("models", []):
        item["status"] = "active" if str(item.get("id")) == str(model.get("id")) else "inactive"
        item["updated_at"] = _now_text()
    registry["updated_at"] = _now_text()
    return model


def remove_model(registry: dict, model_id: str) -> dict | None:
    model = get_model_by_id(registry, model_id)
    if model is None:
        return None
    registry["models"] = [item for item in registry.get("models", []) if str(item.get("id")) != str(model_id)]
    if str(registry.get("active_model_id") or "") == str(model_id):
        registry["active_model_id"] = str(registry["models"][0].get("id", "")) if registry.get("models") else ""
        active = get_active_model(registry)
        if active is not None:
            active["status"] = "active"
    registry["updated_at"] = _now_text()
    return model


def update_model_evaluation(
    registry: dict,
    model_id: str,
    evaluation_type: str,
    dataset: str,
    metrics: dict,
    note: str = "",
    duration_seconds: float | None = None,
) -> dict:
    model = get_model_by_id(registry, model_id)
    if model is None:
        raise KeyError(f"Model not found: {model_id}")

    now = _now_text()
    model["evaluation_count"] = int(model.get("evaluation_count", 0) or 0) + 1
    model["last_eval_at"] = now
    model["last_eval_type"] = str(evaluation_type or "")
    model["last_eval_dataset"] = str(dataset or "")
    model["last_eval_metrics"] = dict(metrics or {})
    model["updated_at"] = now

    evaluation_record = {
        "id": f"eval_{uuid4().hex[:12]}",
        "model_id": str(model.get("id")),
        "model_name": str(model.get("name", "模型")),
        "version": str(model.get("version", "v1.0")),
        "path": str(model.get("path", "")),
        "type": str(evaluation_type or ""),
        "dataset": str(dataset or ""),
        "metrics": dict(metrics or {}),
        "note": str(note or ""),
        "duration_seconds": float(duration_seconds) if duration_seconds is not None else None,
        "created_at": now,
    }
    registry.setdefault("evaluations", []).insert(0, evaluation_record)
    registry["updated_at"] = now
    return evaluation_record
