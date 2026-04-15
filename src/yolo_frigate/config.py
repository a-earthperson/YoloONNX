from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

DEFAULT_EXPORT_CALIBRATION_MAX_SAMPLES = 512
MIN_EXPORT_CALIBRATION_MAX_SAMPLES = 1
MAX_EXPORT_CALIBRATION_MAX_SAMPLES = 4096


@dataclass(frozen=True)
class AppConfig:
    log_level: str
    runtime: str
    label_file: str | None
    model_file: str
    device: str
    confidence_threshold: float
    iou_threshold: float
    frigate_confidence_floor: float
    export_imgsz: int
    export_half: bool
    export_int8: bool
    export_dynamic: bool
    export_nms: bool
    export_batch: int
    export_data: str | None
    export_fraction: float
    export_calibration_max_samples: int
    export_workspace: float | None
    model_cache_dir: str
    enable_save: bool
    save_threshold: str
    save_path: str
    host: str
    port: int


def _bounded_int(name: str, minimum: int, maximum: int):
    def parse(value: str) -> int:
        try:
            parsed = int(value)
        except (
            ValueError
        ) as exc:  # pragma: no cover - argparse formats the final error.
            raise argparse.ArgumentTypeError(f"{name} must be an integer.") from exc
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between {minimum} and {maximum}."
            )
        return parsed

    return parse


def _bounded_float(name: str, minimum: float, maximum: float):
    def parse(value: str) -> float:
        try:
            parsed = float(value)
        except (
            ValueError
        ) as exc:  # pragma: no cover - argparse formats the final error.
            raise argparse.ArgumentTypeError(f"{name} must be a float.") from exc
        if not minimum <= parsed <= maximum:
            raise argparse.ArgumentTypeError(
                f"{name} must be between {minimum} and {maximum}."
            )
        return parsed

    return parse


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="yolo-frigate — https://github.com/a-earthperson/yolo-frigate",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="warning",
        choices=["debug", "info", "warn", "warning", "error", "fatal", "critical"],
        help="Set the logging level (default: warning).",
    )
    parser.add_argument(
        "--runtime",
        type=str,
        default=(os.getenv("YOLO_FRIGATE_RUNTIME") or "auto"),
        choices=["auto", "tensorrt", "openvino", "onnx", "tflite", "edgetpu"],
        help="Native runtime profile. Defaults to YOLO_FRIGATE_RUNTIME, else auto.",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Optional path to the YOLOE class vocabulary file. Entries are passed to set_classes() before export. When omitted for named YOLOE checkpoints, yolo-frigate prefers the matching prompt-free -pf checkpoint.",
    )
    parser.add_argument(
        "--model_file",
        type=str,
        required=True,
        help="Path to a source model (.pt) or runtime artifact (.engine, .onnx, .tflite, *_openvino_model).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Requested device. Common values: cpu, gpu, gpu:<index>, usb, pci.",
    )
    parser.add_argument(
        "--confidence_threshold",
        type=float,
        default=0.25,
        help="Confidence threshold for detection",
    )
    parser.add_argument(
        "--iou_threshold",
        type=float,
        default=0.45,
        help="Intersection over Union (IoU) threshold for detection",
    )
    parser.add_argument(
        "--frigate_confidence_floor",
        type=_bounded_float("frigate_confidence_floor", 0.0, 1.0),
        default=0.0,
        help="Minimum confidence exposed in /detect responses.",
    )
    parser.add_argument(
        "--export_imgsz",
        type=int,
        default=640,
        help="Export image size for lazy .pt conversion.",
    )
    parser.add_argument(
        "--export_half",
        action="store_true",
        help="Enable FP16 export when supported by the selected runtime.",
    )
    parser.add_argument(
        "--export_int8",
        action="store_true",
        help="Enable INT8 export when supported by the selected runtime.",
    )
    parser.add_argument(
        "--export_dynamic",
        action="store_true",
        help="Enable dynamic input shapes during lazy export when supported.",
    )
    parser.add_argument(
        "--export_nms",
        action="store_true",
        help="Embed NMS in the exported model when supported.",
    )
    parser.add_argument(
        "--export_batch",
        type=int,
        default=1,
        help="Maximum batch size to bake into lazily exported artifacts.",
    )
    parser.add_argument(
        "--export_data",
        type=str,
        default=None,
        help="Dataset config used for quantization calibration. When omitted for supported INT8 exports, yolo-frigate bootstraps a cached Open Images V7 validation subset derived from the label map.",
    )
    parser.add_argument(
        "--export_fraction",
        type=float,
        default=1.0,
        help="Dataset fraction used for quantization calibration.",
    )
    parser.add_argument(
        "--export_calibration_max_samples",
        type=_bounded_int(
            "export_calibration_max_samples",
            MIN_EXPORT_CALIBRATION_MAX_SAMPLES,
            MAX_EXPORT_CALIBRATION_MAX_SAMPLES,
        ),
        default=DEFAULT_EXPORT_CALIBRATION_MAX_SAMPLES,
        help=(
            "Maximum number of Open Images calibration samples to materialize "
            "when auto-bootstrapping INT8 export data."
        ),
    )
    parser.add_argument(
        "--export_workspace",
        type=float,
        default=None,
        help="TensorRT workspace size in GiB for lazy export.",
    )
    parser.add_argument(
        "--model_cache_dir",
        type=str,
        default=(
            os.getenv("YOLO_FRIGATE_MODEL_CACHE_DIR") or "/tmp/yolo-frigate-cache"
        ),
        help="Writable directory for lazily exported runtime artifacts. "
        "Default from YOLO_FRIGATE_MODEL_CACHE_DIR.",
    )
    parser.add_argument(
        "--enable_save",
        action="store_true",
        help="Enable saving images and predictions",
    )
    parser.add_argument(
        "--save_threshold",
        type=str,
        default="0.75",
        help="Threshold for saving predictions, can be a float or an expression like deer:0.75,person:0.60-0.75,0.80",
    )
    parser.add_argument(
        "--save_path",
        type=str,
        default="./output",
        help="Folder to save images and predictions",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host address to bind the API server to",
    )
    parser.add_argument(
        "--port", type=int, default=8000, help="Port to bind the API server to"
    )
    return parser


def parse_args(argv: list[str] | None = None) -> AppConfig:
    args = build_arg_parser().parse_args(argv)
    return AppConfig(
        log_level=args.log_level,
        runtime=args.runtime,
        label_file=args.label_file,
        model_file=args.model_file,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        frigate_confidence_floor=args.frigate_confidence_floor,
        export_imgsz=args.export_imgsz,
        export_half=args.export_half,
        export_int8=args.export_int8,
        export_dynamic=args.export_dynamic,
        export_nms=args.export_nms,
        export_batch=args.export_batch,
        export_data=args.export_data,
        export_fraction=args.export_fraction,
        export_calibration_max_samples=args.export_calibration_max_samples,
        export_workspace=args.export_workspace,
        model_cache_dir=args.model_cache_dir,
        enable_save=args.enable_save,
        save_threshold=args.save_threshold,
        save_path=args.save_path,
        host=args.host,
        port=args.port,
    )
