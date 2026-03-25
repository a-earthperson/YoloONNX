from __future__ import annotations

from pathlib import Path

from yolorest.config import AppConfig
from yolorest.detector_backend import DetectorBackend
from yolorest.label import parse_labels
from yolorest.onnx_detector import ONNXDetector


def resolve_backend(config: AppConfig) -> str:
    if config.backend != "auto":
        return config.backend

    suffix = Path(config.model_file).suffix.lower()
    if suffix == ".tflite":
        return "tflite"
    if suffix == ".onnx":
        return "onnx"

    raise ValueError(
        f"Unable to infer backend from model file '{config.model_file}'. "
        "Specify --backend explicitly."
    )


def load_labels(label_file: str | None) -> dict[int, str] | None:
    if label_file is None:
        return None
    return parse_labels(label_file)


def create_detector(config: AppConfig) -> DetectorBackend:
    backend = resolve_backend(config)
    labels = load_labels(config.label_file)

    if backend == "tflite":
        if labels is None:
            raise ValueError("--label_file is required when using the TFLite backend.")

        from yolorest.yolo_flite import YOLOFLite

        return YOLOFLite(
            config.model_file,
            labels,
            config.confidence_threshold,
            config.iou_threshold,
            config.device,
        )

    if backend == "onnx":
        return ONNXDetector(
            config.model_file,
            labels,
            config.confidence_threshold,
            config.iou_threshold,
            config.device,
        )

    raise ValueError(f"Unsupported backend '{backend}'.")
