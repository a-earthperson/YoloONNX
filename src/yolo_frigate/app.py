from __future__ import annotations

import asyncio
import contextlib
import logging
from contextlib import asynccontextmanager
from typing import Annotated

import cv2
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile

from sidecar import install_live_sidecar
from yolo_frigate.detector_backend import DetectorBackend
from yolo_frigate.prediction import Prediction, Predictions
from yolo_frigate.prediction_saver import PredictionItem, PredictionSaver

logger = logging.getLogger(__name__)


def _detect_image_format(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\xff\xd8\xff"):
        return "jpeg"
    if image_bytes.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "gif"
    if image_bytes.startswith(b"BM"):
        return "bmp"
    if image_bytes.startswith((b"II*\x00", b"MM\x00*")):
        return "tiff"
    if (
        len(image_bytes) >= 12
        and image_bytes[0:4] == b"RIFF"
        and image_bytes[8:12] == b"WEBP"
    ):
        return "webp"
    return "unknown"


def _apply_confidence_floor(
    predictions: Predictions, frigate_confidence_floor: float
) -> Predictions:
    if frigate_confidence_floor <= 0:
        return predictions
    return Predictions(
        predictions=[
            Prediction(
                **{
                    **prediction.model_dump(),
                    "confidence": max(prediction.confidence, frigate_confidence_floor),
                }
            )
            for prediction in predictions.predictions
        ],
        success=predictions.success,
    )


def create_app(
    detector: DetectorBackend,
    prediction_saver: PredictionSaver,
    frigate_confidence_floor: float = 0.0,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI):
        task = asyncio.create_task(prediction_saver.process())
        try:
            yield
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

    app = FastAPI(lifespan=lifespan)
    app.state.force_save = False
    install_live_sidecar(app)

    @app.get("/")
    def root():
        logger.debug("Root endpoint accessed.")
        return {"message": "Hello World"}

    @app.get("/health")
    def health():
        return ""

    @app.post("/force_save/{state}")
    def set_force_save(state: bool):
        app.state.force_save = state
        logger.info("Force save set to: %s", app.state.force_save)
        return {"force_save": app.state.force_save}

    @app.post("/detect", response_model=Predictions)
    async def detect_objects(image: Annotated[UploadFile, File()]) -> Predictions:
        logger.debug("Detect endpoint accessed.")
        image_bytes = await image.read()
        binary_content = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(binary_content, cv2.IMREAD_COLOR)
        if img is None:
            logger.error(
                "Failed to decode image upload. filename=%s content_type=%s "
                "format=%s bytes=%s",
                image.filename,
                image.content_type,
                _detect_image_format(image_bytes),
                len(image_bytes),
            )
            raise HTTPException(status_code=400, detail="Invalid image format")

        height, width = img.shape[:2]
        channels = img.shape[2] if img.ndim == 3 else 1
        logger.debug(
            "Received detection image. filename=%s content_type=%s format=%s "
            "bytes=%s width=%s height=%s channels=%s dtype=%s",
            image.filename,
            image.content_type,
            _detect_image_format(image_bytes),
            len(image_bytes),
            width,
            height,
            channels,
            img.dtype,
        )

        predictions = await detector.detect(img)
        detected_labels = (
            ", ".join(
                f"{prediction.label} ({prediction.confidence:.3f})"
                for prediction in predictions.predictions
            )
            or "none"
        )
        logger.debug(
            "Detection completed. Found %s objects: %s",
            len(predictions.predictions),
            detected_labels,
        )
        predictions = _apply_confidence_floor(predictions, frigate_confidence_floor)
        prediction_item = PredictionItem(
            image=image_bytes,
            predictions=predictions,
            forced=app.state.force_save,
        )
        await prediction_saver.add_prediction(prediction_item)
        return predictions

    @app.post("/predict", response_model=Predictions)
    async def predict_objects(image: Annotated[UploadFile, File()]) -> Predictions:
        image_bytes = await image.read()
        binary_content = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(binary_content, cv2.IMREAD_COLOR)
        predictions = await detector.detect(img)
        detected_labels = (
            ", ".join(
                f"{prediction.label} ({prediction.confidence:.3f})"
                for prediction in predictions.predictions
            )
            or "none"
        )
        logger.info(
            "Found %s objects: %s",
            len(predictions.predictions),
            detected_labels,
        )
        return predictions

    return app
