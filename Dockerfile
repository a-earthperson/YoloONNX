ARG TENSORRT_IMAGE="25.01-py3"
FROM nvcr.io/nvidia/tensorrt:${TENSORRT_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv
ENV YOLOREST_RUNTIME=tensorrt
ENV YOLOREST_MODEL_CACHE_DIR=/cache/yolorest
ENV YOLO_CONFIG_DIR=/cache/Ultralytics

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libgl1 \
    libglib2.0-0 \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG UID=1000
ARG GID=1000
RUN mkdir -p /cache/yolorest /cache/Ultralytics && chown -R "${UID}:${GID}" /cache

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --extra tensorrt \
    && uv pip install \
    --python "/app/.venv/bin/python" \
    --index-url https://pypi.org/simple \
    --extra-index-url https://pypi.nvidia.com \
    "tensorrt-cu13>=7.0.0,!=10.1.0"

ENV PATH="/app/.venv/bin:${PATH}"

USER ${UID}:${GID}

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=60s --start-period=60s --retries=10 CMD [ "curl", "--fail", "--silent", "http://localhost:8000/health" ]

ENTRYPOINT ["yolorest"]
