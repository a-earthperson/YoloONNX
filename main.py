from __future__ import annotations

import asyncio
import logging

from uvicorn import Config, Server

from app import create_app
from config import AppConfig, parse_args
from detector_factory import create_detector, resolve_backend
from prediction_saver import PredictionSaver

logger = logging.getLogger(__name__)


def configure_logging(log_level: str) -> None:
    logging.basicConfig(
        level=logging._nameToLevel[log_level.upper()],
        format="%(asctime)s - %(levelname)s - %(message)s",
    )


def build_app(config: AppConfig):
    configure_logging(config.log_level)
    logger.debug("Parsed arguments: %s", config)

    backend = resolve_backend(config)
    detector = create_detector(config)
    logger.info("Initialized %s detector successfully.", backend.upper())

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


if __name__ == "__main__":
    asyncio.run(main())