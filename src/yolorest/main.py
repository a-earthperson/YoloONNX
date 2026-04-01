from __future__ import annotations

import asyncio
import logging

from uvicorn import Config, Server

from yolorest.app import create_app
from yolorest.config import AppConfig, parse_args
from yolorest.detector_factory import create_detector, resolve_runtime
from yolorest.prediction_saver import PredictionSaver

logger = logging.getLogger(__name__)


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=logging._nameToLevel[log_level.upper()],
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def build_app(config: AppConfig):
    configure_logging(config.log_level)
    logger.debug("Parsed arguments: %s", config)

    runtime = resolve_runtime(config)
    detector = create_detector(config)
    logger.info("Initialized %s runtime successfully.", runtime.upper())

    prediction_saver = PredictionSaver(
        config.enable_save,
        config.save_threshold,
        config.save_path,
    )
    return create_app(detector, prediction_saver)


async def main(argv: list[str] | None = None):
    config = parse_args(argv)
    app = build_app(config)
    logger.info("Starting application on %s:%s", config.host, config.port)
    server = Server(Config(app, host=config.host, port=config.port))
    await server.serve()


def run() -> None:
    asyncio.run(main())


if __name__ == "__main__":
    run()
