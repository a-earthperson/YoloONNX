# Yolo-rest

This service exposes a DeepStack-shaped REST API for object detection so it can be used by Frigate and other clients expecting that contract.

The repository now centers on lazy runtime-native export from `.pt` checkpoints. Each Docker image maps to one runtime family:

- `tensorrt`: NVIDIA/TensorRT native `.engine` artifacts
- `openvino`: Intel OpenVINO native `*_openvino_model/` artifacts
- `tflite`: TensorFlow Lite `.tflite` artifacts
- `edgetpu`: Coral EdgeTPU `.tflite` artifacts compiled for EdgeTPU

The REST contract is shared across all runtime profiles:

- `POST /detect` accepts multipart form-data with an `image` file field.
- Success responses follow the `Predictions` schema in `yolorest.prediction`.
- `GET /health` returns an empty string.
- `POST /force_save/{state}` toggles forced save behavior.

## Runtime selection

The primary contract is:

- pass a `.pt` checkpoint to `--model_file`
- let the image decide the runtime via `YOLOREST_RUNTIME`
- let the service lazily export the native artifact on first use

You can still override runtime selection explicitly when running outside Docker:

```bash
uv run yolorest --runtime=tensorrt --device=gpu:0 --model_file=/models/model.pt
uv run yolorest --runtime=openvino --device=cpu --model_file=/models/model.pt
uv run yolorest --runtime=tflite --device=cpu --model_file=/models/model.pt
uv run yolorest --runtime=edgetpu --device=pci --model_file=/models/model.pt
```

The service accepts pre-exported native runtime artifacts:

- TensorRT: `.engine`
- OpenVINO: `*_openvino_model/`
- TFLite / EdgeTPU: `.tflite`

Device semantics are runtime-specific:

- TensorRT: `gpu`, `gpu:0`, `gpu:1`, ...
- OpenVINO: `cpu`, `gpu`, `gpu:0`, `npu`, ...
- TFLite: `cpu`
- EdgeTPU: `usb`, `usb:0`, `pci`, `pci:1`, ...

Notes:

- `--runtime` is the only runtime selector.
- `--label_file` is an override escape hatch rather than a TFLite requirement.

## Lazy export and cache

When `--model_file` points to a `.pt` checkpoint, the service exports lazily on first use:

- TensorRT image: `.pt` -> `.engine`
- OpenVINO image: `.pt` -> `*_openvino_model/`
- TFLite CPU mode: `.pt` -> `.tflite`
- TFLite EdgeTPU mode: `.pt` -> `*_edgetpu.tflite`

Exported artifacts are cached by source hash plus export settings. The cache key includes runtime-sensitive inputs such as:

- runtime profile and export format
- `--export_imgsz`
- `--export_half`
- `--export_int8`
- `--export_dynamic`
- `--export_nms`
- `--export_batch`
- `--export_workspace`
- calibration inputs (`--export_data`, `--export_fraction`)

Important operational notes:

- Docker images default `YOLOREST_MODEL_CACHE_DIR` to `/cache/yolorest`.
- Mount `/cache` as a writable volume when running containers with `read_only: true`.
- `--export_int8` requires `--export_data` so calibration stays deterministic.
- TensorRT INT8 cache entries are hardware-sensitive; export on deployment-class GPUs.
- EdgeTPU export requires an x86 Linux exporter environment.

Useful export flags:

```bash
--export_imgsz=640
--export_half
--export_dynamic
--export_batch=1
--export_workspace=4
--export_int8 --export_data=/models/data.yaml
```

## Runtime-native scope

The runtime layer is now intentionally thin:

- it resolves the runtime profile
- lazily exports `.pt` sources into runtime-native artifacts
- loads the resolved artifact with Ultralytics
- adapts detections into the existing `Predictions` REST contract

The service is intended for Ultralytics detection models and their exported runtime artifacts. It is not a generic executor for arbitrary non-YOLO graphs or custom postprocessing pipelines.

## Docker images

- TFLite / Coral image: built from `Dockerfile.tflite`
- TensorRT image: built from `Dockerfile`
- OpenVINO image: built from `Dockerfile.openvino`

Each image now has a fixed runtime identity through `YOLOREST_RUNTIME`, while the model source can be either a `.pt` checkpoint or an already-exported native artifact.

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
      - ./yolorest/models:/models:ro
      - ./yolorest/cache:/cache
    devices:
      - /dev/apex_0:/dev/apex_0
    command:
      - "--device=pci"
      - "--model_file=/models/yolo11n.pt"
      - "--export_imgsz=320"
```

### TensorRT compose example

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
      - ./yolorest/cache:/cache
    command:
      - "--device=gpu:0"
      - "--model_file=/models/yolo11n.pt"
      - "--export_half"
```

### TensorRT GPU reservation example

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
      - ./yolorest/cache:/cache
    command:
      - "--device=gpu:0"
      - "--model_file=/models/yolo11n.pt"
      - "--export_half"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]
```

### OpenVINO GPU compose example

Docker Intel GPU support requires exposing `/dev/dri` to the container. The OpenVINO Linux wheels are bundled via `openvino`, but the container still needs access to the host GPU device nodes.

```yaml
services:
  yolorest-openvino:
    image: ghcr.io/a-earthperson/yolorest-openvino:${YOLOREST_VERSION}
    container_name: yolorest-openvino
    restart: on-failure
    mem_limit: 1G
    cpus: 2
    read_only: true
    security_opt:
      - "no-new-privileges=true"
    devices:
      - /dev/dri:/dev/dri
    group_add:
      - "${GROUP_RENDER}"
    volumes:
      - ./yolorest/models:/models:ro
      - ./yolorest/cache:/cache
    command:
      - "--device=gpu:0"
      - "--model_file=/models/yolo11n.pt"
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

## Manual export guidance

Ultralytics model source: [ultralytics/assets v8.3.0](https://github.com/ultralytics/assets/releases/tag/v8.3.0)

Lazy export is the default path. Manual export is still useful when you want to prebuild runtime artifacts ahead of deployment.

### TensorRT export

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest \
  yolo export model=/models/<name>.pt format=engine half=True dynamic=True
```

### OpenVINO export

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest \
  yolo export model=/models/<name>.pt format=openvino
```

### TFLite export

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest-cpu \
  yolo export model=/models/<name>.pt format=tflite
```

### EdgeTPU export

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest-cpu \
  yolo export model=/models/<name>.pt format=edgetpu
```

Pre-exported `.engine`, `*_openvino_model/`, and `.tflite` paths can be passed directly to `--model_file`.

## Development

Install [uv](https://docs.astral.sh/uv/). Sync the project (core dependencies plus the `dev` group for tests and formatters), then run the test suite:

```bash
uv sync --group dev
uv run python -m unittest discover -s tests -v
```

Optional inference extras (install only what you need locally):

```bash
uv sync --group dev --extra tflite
uv sync --group dev --extra tensorrt
uv sync --group dev --extra openvino
```

Notes:

- Runtime extras are primarily intended for Linux environments that mirror the Docker images.
- The lazy export path depends on `ultralytics`; TFLite export additionally needs TensorFlow, and native OpenVINO export needs `openvino`.

Format and lint (from the synced environment, or via `uvx` without a local venv):

```bash
uv run ruff check src tests
uv run black src tests
# or: uvx ruff check src tests && uvx black src tests
```

The repo pins the default interpreter with [`.python-version`](.python-version) (3.11, aligned with `Dockerfile`); `uv` respects that for `uv sync` and `uv run`.