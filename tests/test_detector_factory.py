import unittest
from unittest.mock import patch

from yolorest.config import AppConfig
from yolorest.detector_factory import create_detector, resolve_backend


def make_config(**overrides) -> AppConfig:
    values = {
        "log_level": "warning",
        "backend": "auto",
        "label_file": None,
        "model_file": "model.onnx",
        "device": "cpu",
        "confidence_threshold": 0.25,
        "iou_threshold": 0.45,
        "enable_save": False,
        "save_threshold": "0.75",
        "save_path": "./output",
        "host": "0.0.0.0",
        "port": 8000,
    }
    values.update(overrides)
    return AppConfig(**values)


class TestDetectorFactory(unittest.TestCase):
    def test_resolve_backend_from_model_suffix(self):
        self.assertEqual(
            resolve_backend(make_config(model_file="model.tflite")), "tflite"
        )
        self.assertEqual(resolve_backend(make_config(model_file="model.onnx")), "onnx")

    def test_explicit_backend_overrides_suffix(self):
        config = make_config(backend="onnx", model_file="model.bin")
        self.assertEqual(resolve_backend(config), "onnx")

    def test_unknown_model_suffix_requires_explicit_backend(self):
        with self.assertRaises(ValueError):
            resolve_backend(make_config(model_file="model.bin"))

    def test_tflite_backend_requires_label_file(self):
        with self.assertRaises(ValueError):
            create_detector(
                make_config(
                    backend="tflite", model_file="model.tflite", label_file=None
                )
            )

    def test_onnx_backend_is_constructed_with_optional_labels(self):
        config = make_config(
            backend="onnx", model_file="model.onnx", label_file=None, device="cuda:1"
        )
        with patch("yolorest.detector_factory.ONNXDetector") as detector_cls:
            detector_cls.return_value = object()
            detector = create_detector(config)

        detector_cls.assert_called_once_with("model.onnx", None, 0.25, 0.45, "cuda:1")
        self.assertIs(detector, detector_cls.return_value)
