ARG CUDA_IMAGE="nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04"

FROM ${CUDA_IMAGE} AS yolo-frigate-cuda-builder

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy
ENV UV_CACHE_DIR=/root/.cache/uv
ENV UV_PYTHON_DOWNLOADS=never

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    git \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxcb1 \
    libxau6 \
    libxdmcp6 \
    python3 \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv
COPY ../pyproject.toml uv.lock README.md ./
COPY ../src ./src

RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    uv sync --frozen --no-dev --no-editable --extra cuda && \
    python3 - <<'PY'
from pathlib import Path
import subprocess

site_packages = next(Path("/app/.venv/lib").glob("python3.*/site-packages"))
provider = site_packages / "onnxruntime" / "capi" / "libonnxruntime_providers_cuda.so"
ldd = subprocess.run(
    ["ldd", str(provider)],
    check=True,
    capture_output=True,
    text=True,
).stdout
missing = [line.strip() for line in ldd.splitlines() if "not found" in line]
if missing:
    raise SystemExit(
        "Missing CUDA Execution Provider dependencies:\n" + "\n".join(missing)
    )
PY

FROM ${CUDA_IMAGE} AS yolo-frigate-cuda

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV YOLO_FRIGATE_RUNTIME=onnx
ENV YOLO_FRIGATE_MODEL_CACHE_DIR=/cache/yolo-frigate
ENV YOLO_CONFIG_DIR=/cache/Ultralytics
ENV PATH="/app/.venv/bin:${PATH}"
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV UV_PYTHON_DOWNLOADS=never

RUN --mount=target=/var/lib/apt/lists,type=cache,sharing=locked \
    --mount=target=/var/cache/apt,type=cache,sharing=locked \
    apt-get update && \
    apt-get install -y --no-install-recommends \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libx11-6 \
    libxcb1 \
    libxau6 \
    libxdmcp6 \
    python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN mkdir -p /cache/yolo-frigate /cache/Ultralytics /models

COPY --from=yolo-frigate-cuda-builder /app/.venv /app/.venv
COPY ../labelmap.txt /models/

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=60s --start-period=60s --retries=10 CMD [ "python3", "-c", "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=10)" ]

ENTRYPOINT ["yolo-frigate"]
