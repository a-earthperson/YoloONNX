ARG TENSORRT_IMAGE="25.01-py3"
FROM nvcr.io/nvidia/tensorrt:${TENSORRT_IMAGE}

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    python3 \
    python3-pip \
    python3-venv \
    python-is-python3 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ARG UID=10001
RUN useradd \
    --uid "${UID}" \
    --create-home \
    --shell /usr/sbin/nologin \
    appuser

COPY pyproject.toml uv.lock README.md ./
COPY src ./src

RUN uv sync --frozen --no-dev --extra onnx

ENV PATH="/app/.venv/bin:${PATH}"

USER appuser

EXPOSE 8000

HEALTHCHECK --interval=60s --timeout=60s --start-period=60s --retries=10 CMD [ "curl", "--fail", "--silent", "http://localhost:8000/health" ]

ENTRYPOINT ["yolorest", "--backend=onnx"]
