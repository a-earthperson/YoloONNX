import sys
import types
import unittest
import unittest.mock
import os

import numpy as np

from yolorest.onnx_detector import ONNXDetector


class FakeModelMeta:
    def __init__(self, metadata):
        self.custom_metadata_map = metadata


class FakeInput:
    def __init__(self, name="images", shape=None):
        self.name = name
        self.shape = shape or [1, 3, 640, 640]


class FakeOutput:
    def __init__(self, name="output0"):
        self.name = name


class FakeSession:
    instances = []

    def __init__(self, model_file, providers=None):
        self.model_file = model_file
        self.providers = providers
        self.run_calls = []
        self.metadata = {"names": "{0: 'person', 1: 'dog'}"}
        self.input = FakeInput()
        self.output = FakeOutput()
        FakeSession.instances.append(self)

    def get_inputs(self):
        return [self.input]

    def get_outputs(self):
        return [self.output]

    def get_modelmeta(self):
        return FakeModelMeta(self.metadata)

    def run(self, output_names, inputs):
        self.run_calls.append((output_names, inputs))
        detections = np.array(
            [[[20.0], [30.0], [20.0], [20.0], [0.1], [0.85]]],
            dtype=np.float32,
        )
        return [detections]


class TestONNXDetector(unittest.TestCase):
    def tearDown(self):
        FakeSession.instances.clear()

    def test_detect_maps_standard_yolo_outputs_into_contract(self):
        onnxruntime_module = types.SimpleNamespace(
            get_available_providers=lambda: [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            InferenceSession=FakeSession,
        )

        with unittest.mock.patch.dict(
            sys.modules,
            {"onnxruntime": onnxruntime_module},
        ), unittest.mock.patch.dict(
            os.environ,
            {
                "TRT_ENGINE_CACHE_ENABLE": "true",
                "TRT_ENGINE_CACHE_PATH": "/cache/trt",
                "TRT_TIMING_CACHE_ENABLE": "true",
                "TRT_TIMING_CACHE_PATH": "/cache/timing",
                "USE_FP16": "true",
                "TRT_BUILD_HEURISTICS_ENABLE": "true",
                "TRT_BUILDER_OPTIMIZATION_LEVEL": "1",
                "TRT_MAX_PARTITION_ITERATIONS": "5",
                "TRT_MIN_SUBGRAPH_SIZE": "10",
            },
            clear=False,
        ):
            detector = ONNXDetector(
                "model.onnx",
                None,
                conf=0.4,
                iou=0.5,
                device="gpu:1",
                execution_provider="tensorrt",
            )
            predictions = detector.detect(np.zeros((640, 640, 3), dtype=np.uint8))

        self.assertEqual(
            FakeSession.instances[0].providers,
            [
                (
                    "TensorrtExecutionProvider",
                    {
                        "device_id": "1",
                        "trt_engine_cache_enable": True,
                        "trt_engine_cache_path": "/cache/trt",
                        "trt_timing_cache_enable": True,
                        "trt_timing_cache_path": "/cache/timing",
                        "trt_fp16_enable": True,
                        "trt_build_heuristics_enable": True,
                        "trt_builder_optimization_level": 1,
                        "trt_max_partition_iterations": 5,
                        "trt_min_subgraph_size": 10,
                    },
                ),
                ("CUDAExecutionProvider", {"device_id": "1"}),
                "CPUExecutionProvider",
            ],
        )
        self.assertEqual(FakeSession.instances[0].run_calls[0][0], ["output0"])
        self.assertIn("images", FakeSession.instances[0].run_calls[0][1])
        self.assertEqual(len(predictions.predictions), 1)
        self.assertEqual(predictions.predictions[0].label, "dog")
        self.assertAlmostEqual(predictions.predictions[0].confidence, 0.85, places=6)
        self.assertEqual(predictions.predictions[0].y_min, 20.0)
        self.assertEqual(predictions.predictions[0].x_min, 10.0)
        self.assertEqual(predictions.predictions[0].y_max, 40.0)
        self.assertEqual(predictions.predictions[0].x_max, 30.0)
        self.assertTrue(predictions.success)

    def test_explicit_labels_override_model_metadata(self):
        onnxruntime_module = types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=FakeSession,
        )

        with unittest.mock.patch.dict(
            sys.modules,
            {"onnxruntime": onnxruntime_module},
        ):
            detector = ONNXDetector("model.onnx", {1: "vehicle"}, device="cpu")
            predictions = detector.detect(np.zeros((4, 4, 3), dtype=np.uint8))

        self.assertEqual(predictions.predictions[0].label, "vehicle")

    def test_cuda_request_requires_cuda_provider(self):
        onnxruntime_module = types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=FakeSession,
        )

        with unittest.mock.patch.dict(
            sys.modules,
            {"onnxruntime": onnxruntime_module},
        ):
            with self.assertRaises(RuntimeError):
                ONNXDetector("model.onnx", device="gpu", execution_provider="cuda")

    def test_invalid_device_is_rejected(self):
        onnxruntime_module = types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=FakeSession,
        )

        with unittest.mock.patch.dict(
            sys.modules,
            {"onnxruntime": onnxruntime_module},
        ):
            with self.assertRaises(ValueError):
                ONNXDetector("model.onnx", device="cuda")

    def test_gpu_defaults_to_tensorrt_provider(self):
        onnxruntime_module = types.SimpleNamespace(
            get_available_providers=lambda: [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            InferenceSession=FakeSession,
        )

        with unittest.mock.patch.dict(sys.modules, {"onnxruntime": onnxruntime_module}):
            ONNXDetector("model.onnx", device="gpu:0")

        self.assertEqual(FakeSession.instances[0].providers[0][0], "TensorrtExecutionProvider")

    def test_cuda_execution_provider_can_be_requested_explicitly(self):
        onnxruntime_module = types.SimpleNamespace(
            get_available_providers=lambda: [
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            InferenceSession=FakeSession,
        )

        with unittest.mock.patch.dict(sys.modules, {"onnxruntime": onnxruntime_module}):
            ONNXDetector("model.onnx", device="gpu:2", execution_provider="cuda")

        self.assertEqual(
            FakeSession.instances[0].providers,
            [("CUDAExecutionProvider", {"device_id": "2"}), "CPUExecutionProvider"],
        )

    def test_invalid_integer_env_var_is_rejected(self):
        onnxruntime_module = types.SimpleNamespace(
            get_available_providers=lambda: [
                "TensorrtExecutionProvider",
                "CUDAExecutionProvider",
                "CPUExecutionProvider",
            ],
            InferenceSession=FakeSession,
        )

        with unittest.mock.patch.dict(
            sys.modules,
            {"onnxruntime": onnxruntime_module},
        ), unittest.mock.patch.dict(
            os.environ,
            {"TRT_BUILDER_OPTIMIZATION_LEVEL": "fast"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                ONNXDetector("model.onnx", device="gpu:0", execution_provider="tensorrt")

    def test_end2end_outputs_are_supported(self):
        class EndToEndSession(FakeSession):
            def run(self, output_names, inputs):
                self.run_calls.append((output_names, inputs))
                return [
                    np.array(
                        [[[10.0, 20.0, 30.0, 40.0, 0.9, 1.0]]],
                        dtype=np.float32,
                    )
                ]

        onnxruntime_module = types.SimpleNamespace(
            get_available_providers=lambda: ["CPUExecutionProvider"],
            InferenceSession=EndToEndSession,
        )

        with unittest.mock.patch.dict(sys.modules, {"onnxruntime": onnxruntime_module}):
            detector = ONNXDetector("model.onnx", device="cpu")
            predictions = detector.detect(np.zeros((64, 64, 3), dtype=np.uint8))

        self.assertEqual(predictions.predictions[0].label, "dog")
