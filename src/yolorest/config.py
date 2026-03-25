from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass(frozen=True)
class AppConfig:
    log_level: str
    backend: str
    label_file: str | None
    model_file: str
    device: str
    confidence_threshold: float
    iou_threshold: float
    enable_save: bool
    save_threshold: str
    save_path: str
    host: str
    port: int


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="YOLO Rest Application")
    parser.add_argument(
        "--log_level",
        type=str,
        default="warning",
        choices=["debug", "info", "warn", "warning", "error", "fatal", "critical"],
        help="Set the logging level (default: warning).",
    )
    parser.add_argument(
        "--backend",
        type=str,
        default="auto",
        choices=["auto", "tflite", "onnx"],
        help="Inference backend. Defaults to inferring from --model_file.",
    )
    parser.add_argument(
        "--label_file",
        type=str,
        default=None,
        help="Optional path to the label file. Required for TFLite models and overrides embedded ONNX labels.",
    )
    parser.add_argument(
        "--model_file", type=str, required=True, help="Path to the model file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Device to run the model on. TFLite: cpu/usb/pci. ONNX: cpu/cuda/cuda:<index>.",
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
        backend=args.backend,
        label_file=args.label_file,
        model_file=args.model_file,
        device=args.device,
        confidence_threshold=args.confidence_threshold,
        iou_threshold=args.iou_threshold,
        enable_save=args.enable_save,
        save_threshold=args.save_threshold,
        save_path=args.save_path,
        host=args.host,
        port=args.port,
    )
