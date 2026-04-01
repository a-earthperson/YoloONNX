from __future__ import annotations

from yolorest.config import AppConfig
from yolorest.detector_backend import DetectorBackend
from yolorest.label import parse_labels
from yolorest.model_artifact import ModelArtifactManager
from yolorest.runtime_profile import resolve_runtime_profile
from yolorest.ultralytics_detector import UltralyticsDetector


def resolve_runtime(config: AppConfig) -> str:
    return resolve_runtime_profile(config).name


def load_labels(label_file: str | None) -> dict[int, str] | None:
    if label_file is None:
        return None
    return parse_labels(label_file)


def create_detector(
    config: AppConfig,
    artifact_manager: ModelArtifactManager | None = None,
) -> DetectorBackend:
    runtime_profile = resolve_runtime_profile(config)
    labels = load_labels(config.label_file)
    resolved_artifact = (artifact_manager or ModelArtifactManager()).resolve(
        config, runtime_profile
    )

    return UltralyticsDetector(
        resolved_artifact.path,
        runtime_profile.name,
        labels,
        config.confidence_threshold,
        config.iou_threshold,
        config.device,
    )
