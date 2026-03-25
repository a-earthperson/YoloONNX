from __future__ import annotations

import logging
from typing import Any

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
    ):
        self.conf = conf
        self.iou = iou
        self.requested_device = device
        self.predict_device = self._normalize_device(device)

        try:
            import onnxruntime as ort
        except ImportError as exc:
            raise RuntimeError(
                "The ONNX backend requires onnxruntime/onnxruntime-gpu to be installed."
            ) from exc

        available_providers = ort.get_available_providers()
        logger.info("ONNX Runtime providers available: %s", available_providers)
        if (
            self.predict_device != "cpu"
            and "CUDAExecutionProvider" not in available_providers
        ):
            raise RuntimeError(
                "CUDA was requested for the ONNX backend, but CUDAExecutionProvider is not available. "
                f"Available providers: {available_providers}"
            )

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError(
                "The ONNX backend requires the ultralytics package to be installed."
            ) from exc

        logger.info(
            "Initializing ONNX detector with requested device '%s' (Ultralytics device '%s').",
            self.requested_device,
            self.predict_device,
        )
        self.model = YOLO(model)
        self.labels = labels or self._normalize_labels(
            getattr(self.model, "names", None)
        )
        if not self.labels:
            raise ValueError(
                "No labels were provided and the ONNX model does not expose embedded names metadata."
            )

    def _normalize_device(self, device: str) -> str:
        if device == "cpu":
            return "cpu"
        if device == "cuda":
            return "0"
        if device.startswith("cuda:"):
            index = device.split(":", maxsplit=1)[1]
            if not index.isdigit():
                raise ValueError(
                    f"Invalid CUDA device '{device}'. Expected cuda or cuda:<index>."
                )
            return index
        if device.isdigit():
            return device
        raise ValueError(
            f"Invalid ONNX device '{device}'. Supported values are cpu, cuda, or cuda:<index>."
        )

    def _normalize_labels(self, labels: Any) -> dict[int, str]:
        if labels is None:
            return {}
        if isinstance(labels, dict):
            return {int(class_id): str(name) for class_id, name in labels.items()}
        if isinstance(labels, list):
            return {index: str(name) for index, name in enumerate(labels)}
        return {}

    def _to_numpy(self, value: Any) -> np.ndarray:
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        return np.asarray(value)

    def detect(self, img: np.ndarray) -> Predictions:
        results = self.model.predict(
            source=img,
            conf=self.conf,
            iou=self.iou,
            device=self.predict_device,
            verbose=False,
        )

        predictions = Predictions(predictions=[], success=True)
        for result in results:
            boxes = getattr(result, "boxes", None)
            if boxes is None or len(boxes) == 0:
                continue

            xyxy = self._to_numpy(boxes.xyxy)
            class_ids = self._to_numpy(boxes.cls).astype(int)
            scores = self._to_numpy(boxes.conf).astype(float)

            for box, class_id, score in zip(xyxy, class_ids, scores):
                x_min, y_min, x_max, y_max = box.tolist()
                label = self.labels.get(class_id, str(class_id))
                predictions.predictions.append(
                    Prediction(
                        label=label,
                        confidence=float(score),
                        y_min=float(y_min),
                        x_min=float(x_min),
                        y_max=float(y_max),
                        x_max=float(x_max),
                    )
                )
        return predictions
