import sys
import types
import unittest

import numpy as np

from onnx_detector import ONNXDetector


class FakeTensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class FakeBoxes:
    def __init__(self):
        self.xyxy = FakeTensor([[10.0, 20.0, 30.0, 40.0]])
        self.cls = FakeTensor([1])
        self.conf = FakeTensor([0.85])

    def __len__(self):
        return len(self.conf.values)


class FakeResult:
    def __init__(self):
        self.boxes = FakeBoxes()


class FakeYOLO:
    instances = []

    def __init__(self, model_file):
        self.model_file = model_file
        self.names = {0: "person", 1: "dog"}
        self.predict_kwargs = None
        FakeYOLO.instances.append(self)

    def predict(self, **kwargs):
        self.predict_kwargs = kwargs
        return [FakeResult()]


class TestONNXDetector(unittest.TestCase):
    def tearDown(self):
        FakeYOLO.instances.clear()

    def test_detect_maps_ultralytics_results_into_contract(self):
        ultralytics_module = types.SimpleNamespace(YOLO=FakeYOLO)
        onnxruntime_module = types.SimpleNamespace(
            get_available_providers=lambda: ["CUDAExecutionProvider", "CPUExecutionProvider"]
        )

        with unittest.mock.patch.dict(
            sys.modules,
            {"ultralytics": ultralytics_module, "onnxruntime": onnxruntime_module},
        ):
            detector = ONNXDetector("model.onnx", None, conf=0.4, iou=0.5, device="cuda:1")
            predictions = detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))

        self.assertEqual(FakeYOLO.instances[0].predict_kwargs["device"], "1")
        self.assertEqual(FakeYOLO.instances[0].predict_kwargs["conf"], 0.4)
        self.assertEqual(FakeYOLO.instances[0].predict_kwargs["iou"], 0.5)
        self.assertEqual(
            predictions.model_dump(),
            {
                "predictions": [
                    {
                        "label": "dog",
                        "confidence": 0.85,
                        "y_min": 20.0,
                        "x_min": 10.0,
                        "y_max": 40.0,
                        "x_max": 30.0,
                    }
                ],
                "success": True,
            },
        )

    def test_explicit_labels_override_model_metadata(self):
        ultralytics_module = types.SimpleNamespace(YOLO=FakeYOLO)
        onnxruntime_module = types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])

        with unittest.mock.patch.dict(
            sys.modules,
            {"ultralytics": ultralytics_module, "onnxruntime": onnxruntime_module},
        ):
            detector = ONNXDetector("model.onnx", {1: "vehicle"}, device="cpu")
            predictions = detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))

        self.assertEqual(predictions.predictions[0].label, "vehicle")

    def test_cuda_request_requires_cuda_provider(self):
        ultralytics_module = types.SimpleNamespace(YOLO=FakeYOLO)
        onnxruntime_module = types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])

        with unittest.mock.patch.dict(
            sys.modules,
            {"ultralytics": ultralytics_module, "onnxruntime": onnxruntime_module},
        ):
            with self.assertRaises(RuntimeError):
                ONNXDetector("model.onnx", device="cuda")

    def test_invalid_device_is_rejected(self):
        ultralytics_module = types.SimpleNamespace(YOLO=FakeYOLO)
        onnxruntime_module = types.SimpleNamespace(get_available_providers=lambda: ["CPUExecutionProvider"])

        with unittest.mock.patch.dict(
            sys.modules,
            {"ultralytics": ultralytics_module, "onnxruntime": onnxruntime_module},
        ):
            with self.assertRaises(ValueError):
                ONNXDetector("model.onnx", device="gpu")
