# Yolo-rest

This service exposes a DeepStack-shaped REST API for object detection so it can be used by Frigate and other clients expecting that contract.

The repository now supports two inference backends:

- `tflite`: `.tflite` models on CPU or Coral EdgeTPU.
- `onnx`: `.onnx` models through Ultralytics + ONNX Runtime on CPU or NVIDIA CUDA.

The REST contract is shared across both backends:

- `POST /detect` accepts multipart form-data with an `image` file field.
- Success responses follow the `Predictions` schema from `prediction.py`.
- `GET /health` returns an empty string.
- `POST /force_save/{state}` toggles forced save behavior.

## Backend selection

The service can infer the backend from `--model_file`, or you can specify it explicitly:

```bash
python main.py --backend=tflite --model_file=/models/model.tflite --label_file=/models/labels.txt --device=cpu
python main.py --backend=onnx --model_file=/models/model.onnx --device=cpu
python main.py --backend=onnx --model_file=/models/model.onnx --device=cuda
```

Device semantics are backend-specific:

- `tflite`: `cpu`, `usb`, `usb:0`, `usb:1`, `pci`, `pci:1`, `pci:2`, ...
- `onnx`: `cpu`, `cuda`, `cuda:0`, `cuda:1`, ...

For ONNX, `--label_file` is optional. If omitted, the service uses embedded model names when available. A provided label file overrides embedded metadata.

## Supported ONNX scope

The ONNX backend is intentionally thin and delegates model handling to Ultralytics where possible.

Version 1 is intended for:

- detection-focused `.onnx` models that Ultralytics can load with `YOLO(<model>.onnx)`,
- models whose outputs resolve to standard boxes / class ids / confidences through Ultralytics,
- CPU and NVIDIA CUDA execution through ONNX Runtime.

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
    image: ghcr.io/anderssonpeter/yolorest-onnx:${YOLOREST_VERSION}
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
    image: ghcr.io/anderssonpeter/yolorest-onnx:${YOLOREST_VERSION}
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
      - "--device=cuda:0"
      - "--model_file=/models/yolo11n.onnx"
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

- `nms=False` keeps postprocessing behavior closest to standard Ultralytics detection exports.
- `nms=True` may also work if Ultralytics can normalize the exported model correctly, but it should be validated against your specific model.
- If you need dynamic image sizes, export with `dynamic=True` and validate latency and output behavior in your target environment.

## Development

Create a virtualenv and install dev dependencies for the HTTP and contract tests:

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements-dev.txt
.venv/bin/python -m unittest discover -s tests -v
```