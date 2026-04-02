from __future__ import annotations

from typing import Protocol

import numpy as np

from yolo_frigate.prediction import Predictions


class DetectorBackend(Protocol):
    def detect(self, img: np.ndarray) -> Predictions: ...
