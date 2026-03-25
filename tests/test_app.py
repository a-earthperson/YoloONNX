import asyncio
import unittest

import cv2
import numpy as np
from fastapi.testclient import TestClient

from yolorest.app import create_app
from yolorest.prediction import Prediction, Predictions


class FakeDetector:
    def detect(self, img: np.ndarray) -> Predictions:
        return Predictions(
            predictions=[
                Prediction(
                    label="person",
                    confidence=0.9,
                    y_min=1.0,
                    x_min=2.0,
                    y_max=3.0,
                    x_max=4.0,
                )
            ],
            success=True,
        )


class RecordingPredictionSaver:
    def __init__(self):
        self.items = []

    async def add_prediction(self, item):
        self.items.append(item)

    async def process(self):
        while True:
            await asyncio.sleep(3600)


class TestApp(unittest.TestCase):
    def _image_bytes(self) -> bytes:
        success, encoded = cv2.imencode(".jpg", np.zeros((8, 8, 3), dtype=np.uint8))
        self.assertTrue(success)
        return encoded.tobytes()

    def test_detect_contract_shape(self):
        saver = RecordingPredictionSaver()
        app = create_app(FakeDetector(), saver)

        with TestClient(app) as client:
            response = client.post(
                "/detect",
                files={"image": ("frame.jpg", self._image_bytes(), "image/jpeg")},
            )

        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {
                "predictions": [
                    {
                        "label": "person",
                        "confidence": 0.9,
                        "y_min": 1.0,
                        "x_min": 2.0,
                        "y_max": 3.0,
                        "x_max": 4.0,
                    }
                ],
                "success": True,
            },
        )
        self.assertEqual(len(saver.items), 1)
        self.assertFalse(saver.items[0].forced)

    def test_invalid_image_returns_http_400(self):
        saver = RecordingPredictionSaver()
        app = create_app(FakeDetector(), saver)

        with TestClient(app) as client:
            response = client.post(
                "/detect",
                files={
                    "image": (
                        "invalid.bin",
                        b"not-an-image",
                        "application/octet-stream",
                    )
                },
            )

        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.json(), {"detail": "Invalid image format"})

    def test_force_save_flag_is_propagated(self):
        saver = RecordingPredictionSaver()
        app = create_app(FakeDetector(), saver)

        with TestClient(app) as client:
            toggle_response = client.post("/force_save/true")
            detect_response = client.post(
                "/detect",
                files={"image": ("frame.jpg", self._image_bytes(), "image/jpeg")},
            )

        self.assertEqual(toggle_response.status_code, 200)
        self.assertEqual(toggle_response.json(), {"force_save": True})
        self.assertEqual(detect_response.status_code, 200)
        self.assertTrue(saver.items[0].forced)
