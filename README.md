# Yolo-rest

This service exposes a DeepStack-shaped REST API for object detection so it can be used by Frigate and other clients expecting that contract.

The repository now supports two inference backends:

- `tflite`: `.tflite` models on CPU or Coral EdgeTPU.
- `onnx`: `.onnx` models through ONNX Runtime on CPU or NVIDIA GPU, with TensorRT preferred by default for GPU execution.

The REST contract is shared across both backends:

- `POST /detect` accepts multipart form-data with an `image` file field.
- Success responses follow the `Predictions` schema in `yolorest.prediction`.
- `GET /health` returns an empty string.
- `POST /force_save/{state}` toggles forced save behavior.

## Backend selection

The service can infer the backend from `--model_file`, or you can specify it explicitly:

```bash
uv run yolorest --backend=tflite --model_file=/models/model.tflite --label_file=/models/labels.txt --device=cpu
uv run yolorest --backend=onnx --model_file=/models/model.onnx --device=cpu
uv run yolorest --backend=onnx --model_file=/models/model.onnx --device=gpu
uv run yolorest --backend=onnx --model_file=/models/model.onnx --device=gpu:1 --execution_provider=cuda
```

Device semantics are backend-specific:

- `tflite`: `cpu`, `usb`, `usb:0`, `usb:1`, `pci`, `pci:1`, `pci:2`, ...
- `onnx`: `cpu`, `gpu`, `gpu:0`, `gpu:1`, ...

For ONNX, `--execution_provider` supports `cpu`, `cuda`, and `tensorrt`. If omitted, it defaults to `tensorrt`. When `--device=cpu`, the service always uses `CPUExecutionProvider`.

For ONNX, `--label_file` is optional. If omitted, the service uses embedded model names when available. A provided label file overrides embedded metadata.

TensorRT tuning is controlled via environment variables instead of CLI flags:

- `TRT_ENGINE_CACHE_ENABLE=true|false`
- `TRT_ENGINE_CACHE_PATH=/tmp/ort_trt_cache`
- `TRT_TIMING_CACHE_ENABLE=true|false`
- `TRT_TIMING_CACHE_PATH=/tmp/ort_trt_cache`
- `TRT_BUILD_HEURISTICS_ENABLE=true|false`
- `TRT_BUILDER_OPTIMIZATION_LEVEL=<int>`
- `TRT_MAX_PARTITION_ITERATIONS=<int>`
- `TRT_MIN_SUBGRAPH_SIZE=<int>`
- `USE_FP16=true|false`
- `USE_INT8=true|false`

Recommended TensorRT startup defaults when first-build latency matters:

```env
TRT_ENGINE_CACHE_ENABLE=true
TRT_ENGINE_CACHE_PATH=/config/ort_trt_cache
TRT_TIMING_CACHE_ENABLE=true
TRT_TIMING_CACHE_PATH=/config/ort_trt_cache
TRT_BUILD_HEURISTICS_ENABLE=true
TRT_BUILDER_OPTIMIZATION_LEVEL=1
TRT_MAX_PARTITION_ITERATIONS=5
TRT_MIN_SUBGRAPH_SIZE=10
USE_FP16=true
USE_INT8=false
```

Notes:

- Engine cache avoids rebuilding the TensorRT engine on subsequent starts.
- Timing cache reduces tactic search time during future engine builds.
- Lower builder optimization levels generally reduce compile time at the cost of some runtime performance.
- `USE_INT8` should normally stay `false` unless you have a model/export path that is explicitly validated for TensorRT INT8.

## Supported ONNX scope

The ONNX backend is intentionally thin and executes models directly with ONNX Runtime.

Version 1 is intended for:

- detection-focused `.onnx` models exported from Ultralytics YOLO,
- standard detection exports whose outputs follow the usual YOLO `(batch, 4 + classes, num_predictions)` layout,
- common end-to-end exports that emit `[x1, y1, x2, y2, score, class_id]`,
- CPU, CUDA, and TensorRT execution through ONNX Runtime.

It is not a generic executor for arbitrary non-YOLO ONNX graphs or every possible custom postprocessing topology.

## Docker images

- TFLite / Coral image: built from `Dockerfile`
- ONNX image: built from `Dockerfile.onnx`

The ONNX image is a single image family that can run on CPU or NVIDIA GPU depending on `--device` and host runtime availability.

### TFLite / Coral compose example

```yaml
networks:
  yolorest:
    driver: bridge
    internal: true
    driver_opts:
      com.docker.network.bridge.name: yolorest-dsp

services:
  yolorest:
    image: ghcr.io/anderssonpeter/yolorest:${YOLOREST_VERSION}
    container_name: yolorest
    restart: on-failure
    mem_limit: 256M
    cpus: 0.5
    user: "${USER_SURVEILLANCE}:${GROUP_SURVEILLANCE}"
    read_only: true
    group_add:
      - "${GROUP_CORAL}"
    security_opt:
      - "no-new-privileges=true"
    networks:
      yolorest:
    environment:
      - TZ=Europe/Stockholm
    volumes:
      - /etc/localtime:/etc/localtime:ro
      - ./yolorest/models:/models
    devices:
      - /dev/apex_0:/dev/apex_0
    command:
      - "--backend=tflite"
      - "--device=pci"
      - "--label_file=/models/labelmap_yolov8.txt"
      - "--model_file=/models/yolov8m_320_edgetpu.tflite"
```

### ONNX CPU compose example

```yaml
services:
  yolorest-onnx:
    image: ghcr.io/a-earthperson/yolorest-onnx:${YOLOREST_VERSION}
    container_name: yolorest-onnx
    restart: on-failure
    mem_limit: 1G
    cpus: 2
    read_only: true
    security_opt:
      - "no-new-privileges=true"
    volumes:
      - ./yolorest/models:/models:ro
    command:
      - "--backend=onnx"
      - "--device=cpu"
      - "--model_file=/models/yolo11n.onnx"
```

### ONNX GPU compose example

Docker GPU support requires NVIDIA drivers, NVIDIA Container Toolkit, and Compose GPU reservations.

```yaml
services:
  yolorest-onnx:
    image: ghcr.io/a-earthperson/yolorest-onnx:${YOLOREST_VERSION}
    container_name: yolorest-onnx
    restart: on-failure
    mem_limit: 2G
    read_only: true
    security_opt:
      - "no-new-privileges=true"
    volumes:
      - ./yolorest/models:/models:ro
    command:
      - "--backend=onnx"
      - "--device=gpu:0"
      - "--model_file=/models/yolo11n.onnx"
    environment:
      - TRT_ENGINE_CACHE_ENABLE=true
      - TRT_ENGINE_CACHE_PATH=/tmp/ort_trt_cache
      - TRT_TIMING_CACHE_ENABLE=true
      - TRT_TIMING_CACHE_PATH=/tmp/ort_trt_cache
      - TRT_BUILD_HEURISTICS_ENABLE=true
      - TRT_BUILDER_OPTIMIZATION_LEVEL=1
      - TRT_MAX_PARTITION_ITERATIONS=5
      - TRT_MIN_SUBGRAPH_SIZE=10
      - USE_FP16=true
      - USE_INT8=false
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

## Frigate config

```yaml
detectors:
  yolo-rest:
    type: deepstack
    api_url: http://yolorest:8000/detect
    api_timeout: 0.18

model:
  labelmap_path: /models/labelmap_yolov8.txt
```

## Model export guidance

Ultralytics model source: [ultralytics/assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0)

### TFLite / EdgeTPU export

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest-cpu \
  yolo export model=/models/<name>.pt format=edgetpu
```

### ONNX export

Recommended starting point for the ONNX backend:

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest-cpu \
  yolo export model=/models/<name>.pt format=onnx simplify=True dynamic=False nms=False
```

Notes:

- `nms=False` is the primary supported path and keeps postprocessing behavior closest to standard YOLO detection exports.
- `nms=True` may work for common end-to-end exports, but it should still be validated against your specific model.
- If you need dynamic image sizes, export with `dynamic=True` and validate latency and output behavior in your target environment.

## Development

Install [uv](https://docs.astral.sh/uv/). Sync the project (core dependencies plus the `dev` group for tests and formatters), then run the test suite:

```bash
uv sync --group dev
uv run python -m unittest discover -s tests -v
```

Optional inference extras (install only what you need locally):

```bash
uv sync --group dev --extra tflite
uv sync --group dev --extra onnx
```

Format and lint (from the synced environment, or via `uvx` without a local venv):

```bash
uv run ruff check src tests
uv run black src tests
# or: uvx ruff check src tests && uvx black src tests
```

The repo pins the default interpreter with [`.python-version`](.python-version) (3.11, aligned with `Dockerfile`); `uv` respects that for `uv sync` and `uv run`.