from __future__ import annotations

import ast
import logging
import os
import time
from typing import Any

import cv2
import numpy as np

from yolorest.prediction import Prediction, Predictions

logger = logging.getLogger(__name__)


class ONNXDetector:
    def __init__(
        self,
        model: str,
        labels: dict[int, str] | None = None,
        conf: float = 0.25,
        iou: float = 0.45,
        device: str = "cpu",
        execution_provider: str = "tensorrt",
    ):
        self.conf = conf
        self.iou = iou
        self.requested_device = device
        self.requested_execution_provider = execution_provider
        self.providers = self._build_providers(device, execution_provider)

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "The ONNX backend requires onnxruntime/onnxruntime-gpu to be installed."
            ) from exc

        available_providers = ort.get_available_providers()
        logger.info("ONNX Runtime providers available: %s", available_providers)
        self._validate_available_providers(available_providers)

        logger.info(
            "Initializing ONNX detector with requested device '%s' using providers '%s'.",
            self.requested_device,
            [provider if isinstance(provider, str) else provider[0] for provider in self.providers],
        )
        self.session = ort.InferenceSession(model, providers=self.providers)
        self.input = self.session.get_inputs()[0]
        self.output_names = [output.name for output in self.session.get_outputs()]
        self.input_layout = self._resolve_input_layout(self.input.shape)
        self.model_height, self.model_width = self._resolve_input_shape(self.input.shape)
        self.labels = labels or self._labels_from_metadata(
            self.session.get_modelmeta().custom_metadata_map
        )
        if not self.labels:
            raise ValueError(
                "No labels were provided and the ONNX model does not expose embedded names metadata."
            )

    def _normalize_device(self, device: str) -> tuple[str, str | None]:
        if device == "cpu":
            return "cpu", None
        if device == "gpu":
            return "gpu", "0"
        if device.startswith("gpu:"):
            index = device.split(":", maxsplit=1)[1]
            if not index.isdigit():
                raise ValueError(
                    f"Invalid GPU device '{device}'. Expected gpu or gpu:<index>."
                )
            return "gpu", index
        raise ValueError(
            f"Invalid ONNX device '{device}'. Supported values are cpu, gpu, or gpu:<index>."
        )

    def _build_providers(
        self, device: str, execution_provider: str
    ) -> list[str | tuple[str, dict[str, str | int | bool]]]:
        device_kind, device_id = self._normalize_device(device)
        if execution_provider == "cpu" or device_kind == "cpu":
            return ["CPUExecutionProvider"]

        if device_id is None:
            raise ValueError("GPU execution requires a gpu device identifier.")

        cuda_options: dict[str, str | int | bool] = {"device_id": device_id}
        providers: list[str | tuple[str, dict[str, str | int | bool]]] = [
            "CPUExecutionProvider"
        ]

        if execution_provider == "cuda":
            providers.insert(0, ("CUDAExecutionProvider", cuda_options))
            return providers

        if execution_provider != "tensorrt":
            raise ValueError(
                f"Invalid execution provider '{execution_provider}'. "
                "Supported values are cpu, cuda, or tensorrt."
            )

        trt_options: dict[str, str | int | bool] = {"device_id": device_id}
        engine_cache_enable = self._env_bool("TRT_ENGINE_CACHE_ENABLE", True)
        if engine_cache_enable:
            trt_options["trt_engine_cache_enable"] = True
            trt_options["trt_engine_cache_path"] = os.getenv(
                "TRT_ENGINE_CACHE_PATH", "/tmp/ort_trt_cache"
            )
        timing_cache_enable = self._env_bool("TRT_TIMING_CACHE_ENABLE", True)
        if timing_cache_enable:
            trt_options["trt_timing_cache_enable"] = True
            trt_options["trt_timing_cache_path"] = os.getenv(
                "TRT_TIMING_CACHE_PATH",
                os.getenv("TRT_ENGINE_CACHE_PATH", "/tmp/ort_trt_cache"),
            )
        if self._env_bool("USE_FP16", False):
            trt_options["trt_fp16_enable"] = True
        if self._env_bool("USE_INT8", False):
            trt_options["trt_int8_enable"] = True
        if self._env_bool("TRT_BUILD_HEURISTICS_ENABLE", False):
            trt_options["trt_build_heuristics_enable"] = True

        builder_optimization_level = self._env_int(
            "TRT_BUILDER_OPTIMIZATION_LEVEL", None
        )
        if builder_optimization_level is not None:
            trt_options["trt_builder_optimization_level"] = builder_optimization_level

        max_partition_iterations = self._env_int("TRT_MAX_PARTITION_ITERATIONS", None)
        if max_partition_iterations is not None:
            trt_options["trt_max_partition_iterations"] = max_partition_iterations

        min_subgraph_size = self._env_int("TRT_MIN_SUBGRAPH_SIZE", None)
        if min_subgraph_size is not None:
            trt_options["trt_min_subgraph_size"] = min_subgraph_size

        providers.insert(0, ("CUDAExecutionProvider", cuda_options))
        providers.insert(0, ("TensorrtExecutionProvider", trt_options))
        return providers

    def _validate_available_providers(self, available_providers: list[str]) -> None:
        for provider in self.providers:
            provider_name = provider if isinstance(provider, str) else provider[0]
            if provider_name not in available_providers:
                raise RuntimeError(
                    f"Requested execution provider '{provider_name}' is not available. "
                    f"Available providers: {available_providers}"
                )

    def _env_bool(self, name: str, default: bool) -> bool:
        value = os.getenv(name)
        if value is None:
            return default
        return value.strip().lower() in {"1", "true", "yes", "on"}

    def _env_int(self, name: str, default: int | None) -> int | None:
        value = os.getenv(name)
        if value is None or value.strip() == "":
            return default
        try:
            return int(value)
        except ValueError as exc:
            raise ValueError(
                f"Environment variable {name} must be an integer, got '{value}'."
            ) from exc

    def _resolve_input_layout(self, shape: list[Any]) -> str:
        if len(shape) != 4:
            raise ValueError(f"Unsupported ONNX input rank {len(shape)}. Expected a 4D tensor.")
        channels_first = shape[1]
        channels_last = shape[3]
        if isinstance(channels_first, int) and channels_first in (1, 3):
            return "nchw"
        if isinstance(channels_last, int) and channels_last in (1, 3):
            return "nhwc"
        return "nchw"

    def _resolve_input_shape(self, shape: list[Any]) -> tuple[int | None, int | None]:
        if self.input_layout == "nchw":
            height, width = shape[2], shape[3]
        else:
            height, width = shape[1], shape[2]
        return self._optional_int(height), self._optional_int(width)

    def _optional_int(self, value: Any) -> int | None:
        return int(value) if isinstance(value, int) and value > 0 else None

    def _labels_from_metadata(self, metadata: dict[str, str]) -> dict[int, str]:
        names = metadata.get("names")
        if names is None:
            return {}
        try:
            parsed = ast.literal_eval(names)
        except (ValueError, SyntaxError):
            logger.warning("Failed to parse ONNX metadata names: %s", names)
            return {}

        if isinstance(parsed, dict):
            return {int(class_id): str(name) for class_id, name in parsed.items()}
        if isinstance(parsed, list):
            return {index: str(name) for index, name in enumerate(parsed)}
        return {}

    def letterbox(
        self, img: np.ndarray, new_shape: tuple[int, int]
    ) -> tuple[np.ndarray, float, tuple[float, float]]:
        start_time = time.time()
        shape = img.shape[:2]
        ratio = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * ratio)), int(round(shape[0] * ratio))
        dw = (new_shape[1] - new_unpad[0]) / 2
        dh = (new_shape[0] - new_unpad[1]) / 2

        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
        )
        logger.debug("letterbox took %.2f ms", (time.time() - start_time) * 1000)
        return img, ratio, (left, top)

    def preprocess(
        self, img: np.ndarray
    ) -> tuple[np.ndarray, float, tuple[float, float], tuple[int, int]]:
        target_height = self.model_height or img.shape[0]
        target_width = self.model_width or img.shape[1]
        letterboxed, ratio, pad = self.letterbox(img, (target_height, target_width))
        x = letterboxed[..., ::-1].astype(np.float32) / 255.0
        if self.input_layout == "nchw":
            x = np.transpose(x, (2, 0, 1))[None]
        else:
            x = x[None]
        return np.ascontiguousarray(x), ratio, pad, (target_height, target_width)

    def _clip_box(
        self, x1: float, y1: float, x2: float, y2: float, img_shape: tuple[int, int, int]
    ) -> tuple[float, float, float, float]:
        width = img_shape[1]
        height = img_shape[0]
        return (
            float(np.clip(x1, 0, width)),
            float(np.clip(y1, 0, height)),
            float(np.clip(x2, 0, width)),
            float(np.clip(y2, 0, height)),
        )

    def _postprocess_standard(
        self,
        img: np.ndarray,
        output: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
    ) -> Predictions:
        if output.ndim == 2:
            output = output[None, ...]
        if output.shape[1] > output.shape[2]:
            output = output.transpose(0, 2, 1)

        predictions = Predictions(predictions=[], success=True)
        for out in output:
            if out.shape[-1] <= 4:
                continue
            scores = out[:, 4:].max(-1)
            keep = scores > self.conf
            if not keep.any():
                continue
            boxes = out[keep, :4].copy()
            scores = scores[keep]
            class_ids = out[keep, 4:].argmax(-1)

            boxes[:, 0] -= boxes[:, 2] / 2
            boxes[:, 1] -= boxes[:, 3] / 2
            boxes[:, 0] = (boxes[:, 0] - pad[0]) / ratio
            boxes[:, 1] = (boxes[:, 1] - pad[1]) / ratio
            boxes[:, 2] /= ratio
            boxes[:, 3] /= ratio

            indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), self.conf, self.iou)
            if len(indices) == 0:
                continue
            for index in np.asarray(indices).flatten():
                left, top, width, height = boxes[index]
                x1, y1, x2, y2 = self._clip_box(
                    left, top, left + width, top + height, img.shape
                )
                predictions.predictions.append(
                    Prediction(
                        label=self.labels.get(int(class_ids[index]), str(class_ids[index])),
                        confidence=float(scores[index]),
                        y_min=y1,
                        x_min=x1,
                        y_max=y2,
                        x_max=x2,
                    )
                )
        return predictions

    def _postprocess_end2end(
        self,
        img: np.ndarray,
        output: np.ndarray,
        ratio: float,
        pad: tuple[float, float],
        input_shape: tuple[int, int],
    ) -> Predictions:
        if output.ndim == 2:
            output = output[None, ...]
        predictions = Predictions(predictions=[], success=True)
        input_width = input_shape[1]
        input_height = input_shape[0]

        for out in output:
            for row in out:
                if row.shape[0] < 6:
                    continue
                x1, y1, x2, y2, score, class_id = row[:6]
                if float(score) < self.conf:
                    continue
                if max(x2, y2) <= 1.5:
                    x1 *= input_width
                    x2 *= input_width
                    y1 *= input_height
                    y2 *= input_height
                x1 = (x1 - pad[0]) / ratio
                x2 = (x2 - pad[0]) / ratio
                y1 = (y1 - pad[1]) / ratio
                y2 = (y2 - pad[1]) / ratio
                x1, y1, x2, y2 = self._clip_box(x1, y1, x2, y2, img.shape)
                predictions.predictions.append(
                    Prediction(
                        label=self.labels.get(int(class_id), str(int(class_id))),
                        confidence=float(score),
                        y_min=y1,
                        x_min=x1,
                        y_max=y2,
                        x_max=x2,
                    )
                )
        return predictions

    def postprocess(
        self,
        img: np.ndarray,
        outputs: list[np.ndarray],
        ratio: float,
        pad: tuple[float, float],
        input_shape: tuple[int, int],
    ) -> Predictions:
        if not outputs:
            return Predictions(predictions=[], success=True)

        output = np.asarray(outputs[0])
        if output.ndim >= 2 and output.shape[-1] in (6, 7):
            return self._postprocess_end2end(img, output, ratio, pad, input_shape)
        return self._postprocess_standard(img, output, ratio, pad)

    def detect(self, img: np.ndarray) -> Predictions:
        x, ratio, pad, input_shape = self.preprocess(img)
        outputs = self.session.run(self.output_names, {self.input.name: x})
        return self.postprocess(img, outputs, ratio, pad, input_shape)
