from __future__ import annotations

import os
import tempfile
from pathlib import Path


def import_ultralytics_yolo():
    _prepare_ultralytics_environment()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is required for lazy export and runtime-native inference."
        ) from exc
    return YOLO


def get_ultralytics_version() -> str | None:
    _prepare_ultralytics_environment()
    try:
        import ultralytics
    except ImportError:
        return None
    return getattr(ultralytics, "__version__", None)


def _prepare_ultralytics_environment() -> None:
    configured = os.getenv("YOLO_CONFIG_DIR")
    if configured and _is_writable_directory(Path(configured)):
        return

    cache_root = os.getenv("YOLOREST_MODEL_CACHE_DIR")
    candidates = []
    if cache_root:
        candidates.append(Path(cache_root).parent / "Ultralytics")
    candidates.append(Path(tempfile.gettempdir()) / "Ultralytics")

    for candidate in candidates:
        if _ensure_writable_directory(candidate):
            os.environ["YOLO_CONFIG_DIR"] = str(candidate)
            return


def _is_writable_directory(path: Path) -> bool:
    return path.is_dir() and os.access(path, os.W_OK | os.X_OK)


def _ensure_writable_directory(path: Path) -> bool:
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    return _is_writable_directory(path)
