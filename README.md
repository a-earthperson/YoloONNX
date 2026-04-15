# yolo-frigate

Source: **[github.com/a-earthperson/yolo-frigate](https://github.com/a-earthperson/yolo-frigate)**.

HTTP object detection for **NVR-style stacks**—most often [Frigate](https://github.com/blakeblackshear/frigate) or anything else that speaks a **DeepStack-compatible** multipart `POST /detect` API. The service wraps Ultralytics YOLO models, optionally **lazy-exporting** Ultralytics `.pt` checkpoints into the runtime backend selected by the chosen image profile on first use.

Typical deployments:

- One container per inference node, with the **image profile** chosen for the target userspace stack (`tensorrt`, `cuda`, `onnx`, `openvino`, or `tflite`).
- Shared **read-only model trees** and **writable export caches** (bind mounts, named volumes, or NFS when workers are spread across machines).
- **Orchestrator placement** so NVIDIA-oriented profiles (`tensorrt`, `cuda`, `onnx`) land on NVIDIA hosts, `openvino` lands on Intel GPU / NPU hosts, and `tflite` lands on CPU or Coral EdgeTPU hosts.

## Image profiles

The Dockerfile stem names the **deployment profile**, not the runtime backend. A profile chooses the base image, native userspace, and dependency bundle. The application runtime is still selected by `YOLO_FRIGATE_RUNTIME` plus model/device resolution inside that profile.

| Profile | Dockerfile | Packaged userspace | Default runtime/backend | Native artifacts | Typical devices |
|---------|------------|--------------------|-------------------------|------------------|-----------------|
| **TensorRT** | [`tensorrt.Dockerfile`](tensorrt.Dockerfile) | NVIDIA TensorRT userspace | `tensorrt` | `.engine` | NVIDIA: `gpu`, `gpu:0`, ... |
| **CUDA** | [`cuda.Dockerfile`](cuda.Dockerfile) | NVIDIA CUDA base image plus ONNX/CUDA stack | `onnx` | `.onnx` | `cpu` or NVIDIA: `gpu`, `gpu:0`, ... |
| **ONNX** | [`onnx.Dockerfile`](onnx.Dockerfile) | Wheel-based ONNX Runtime GPU stack on slim Python | `onnx` | `.onnx` | `cpu` or NVIDIA: `gpu`, `gpu:0`, ... |
| **OpenVINO** | [`openvino.Dockerfile`](openvino.Dockerfile) | Intel OpenVINO + Level Zero userspace | `openvino` | `*_openvino_model/` | CPU; Intel GPU: `gpu`, `gpu:0`, ...; NPU where supported |
| **TFLite** | [`tflite.Dockerfile`](tflite.Dockerfile) | TensorFlow Lite + Coral tooling | `tflite` | `.tflite` | `cpu`; Coral: `usb`, `pci`, ... |

Notes:

- `cuda` and `onnx` are different deployment profiles for the same application runtime backend: both run with `--runtime=onnx`, but `cuda` bakes in an NVIDIA CUDA userspace while `onnx` stays on a slimmer Python base image.
- `tflite` sets `YOLO_FRIGATE_RUNTIME=tflite`, but `.tflite` models resolve to `edgetpu` automatically when `--device` names a Coral accelerator instead of `cpu`.

Release builds publish five images from [`.github/workflows/publish.yml`](.github/workflows/publish.yml). The **image name** is the GitHub `owner/repository` name plus a **profile suffix** that matches the Dockerfile stem (GHCR normalizes names to lowercase):

- `ghcr.io/a-earthperson/yolo-frigate-tensorrt` — from `tensorrt.Dockerfile`
- `ghcr.io/a-earthperson/yolo-frigate-cuda` — from `cuda.Dockerfile`
- `ghcr.io/a-earthperson/yolo-frigate-onnx` — from `onnx.Dockerfile`
- `ghcr.io/a-earthperson/yolo-frigate-openvino` — from `openvino.Dockerfile`
- `ghcr.io/a-earthperson/yolo-frigate-tflite` — from `tflite.Dockerfile`

Forks and private mirrors use their own `owner/repo` in place of `a-earthperson/yolo-frigate`. Version **tags** (for example `v0.1.8`, `latest`) come from the GitHub release via `docker/metadata-action`. You may **retag or mirror** under other names if your registry layout requires it.

## API contract

Shared across all runtime profiles:

- `POST /detect` — multipart form-data with an `image` file field.
- Success responses follow the `Predictions` schema in `yolo_frigate.prediction`.
- `GET /health` — liveness (empty body).
- `POST /force_save/{state}` — toggles forced save behavior for debugging.

## Runtime selection and devices

Inside Docker, the image profile constrains the available native stack, while `YOLO_FRIGATE_RUNTIME` selects the default backend inside that stack. Legacy images may still use `YOLOREST_RUNTIME`; the application reads both. You normally pass a `.pt` checkpoint to `--model_file` and let the selected profile export lazily.

Overrides when running outside Docker:

```bash
uv run yolo-frigate --runtime=tensorrt --device=gpu:0 --model_file=/models/model.pt
uv run yolo-frigate --runtime=onnx --device=cpu --model_file=/models/model.pt
uv run yolo-frigate --runtime=onnx --device=gpu:0 --model_file=/models/model.pt
uv run yolo-frigate --runtime=openvino --device=cpu --model_file=/models/model.pt
uv run yolo-frigate --runtime=tflite --device=cpu --model_file=/models/model.pt
uv run yolo-frigate --runtime=edgetpu --device=pci --model_file=/models/model.pt
```

The installable project is **`yolo-frigate`**; the Python import package is **`yolo_frigate`** (for example `python -m yolo_frigate`).

Pre-exported artifacts can be passed directly to `--model_file`: TensorRT `.engine`, ONNX `.onnx`, OpenVINO `*_openvino_model/`, TFLite / EdgeTPU `.tflite`.

Device strings are runtime-specific (TensorRT: `gpu`, `gpu:0`, ...; ONNX: `cpu`, `gpu`, `gpu:0`, ...; OpenVINO: `cpu`, `gpu`, `npu`, ...; TFLite: `cpu`; EdgeTPU: `usb`, `pci`, ...). `--runtime` is the backend selector; the Dockerfile/profile name is only a packaging choice. `--label_file` overrides embedded class names when needed.

## Lazy export and cache

When `--model_file` is a `.pt` checkpoint, export happens on first inference. Cached outputs live under `YOLO_FRIGATE_MODEL_CACHE_DIR` (default `/cache/yolo-frigate` in images; legacy `YOLOREST_MODEL_CACHE_DIR` and `/cache/yolorest` are still honored). The cache key includes profile, `--export_imgsz`, `--export_half`, `--export_int8`, `--export_dynamic`, `--export_nms`, `--export_batch`, `--export_workspace`, and calibration inputs.

Operational notes:

- Mount a **writable** `/cache` (or custom `YOLO_FRIGATE_MODEL_CACHE_DIR`) whenever the container filesystem is read-only.
- `--export_int8` uses `--export_data` when provided; otherwise TensorRT, OpenVINO, and TFLite bootstrap a cached deterministic Open Images V7 validation subset derived from `--label_file`. `--export_calibration_max_samples` controls the subset size (default `512`, valid range `1..4096`). Labels not present in Open Images are ignored with a warning.
- TensorRT INT8 caches are **GPU-generation sensitive**; export on hardware representative of production.
- EdgeTPU export expects an x86 Linux exporter environment.

Common export flags:

```bash
--export_imgsz=640
--export_half
--export_dynamic
--export_batch=1
--export_workspace=4
--export_calibration_max_samples=512
--export_int8 --export_data=/models/data.yaml
```

## Deployment patterns

### What every container needs

- **`/models`** — readable tree with checkpoints or pre-exported artifacts and optional `labelmap.txt` (or pass `--label_file`).
- **`/cache`** — writable export cache (omit only if you never lazy-export or you redirect `YOLO_FRIGATE_MODEL_CACHE_DIR` elsewhere).
- **Devices** — NVIDIA Container Toolkit + GPU reservation for `tensorrt`, `cuda`, and typically `onnx`; `/dev/dri` (and often `video`/`render` groups) for `openvino`; Coral device nodes for `tflite` when using `edgetpu` devices.
- **Shared memory** — large `tmpfs` on `/dev/shm` can help some workloads when memory pressure appears during export or batching.

### Docker Compose (single host)

Minimal TensorRT service (image name ends with `-tensorrt` for releases from this repo):

```yaml
services:
  yolo-frigate-tensorrt:
    image: ghcr.io/a-earthperson/yolo-frigate-tensorrt:v1.2.3
    restart: on-failure
    read_only: true
    security_opt:
      - "no-new-privileges=true"
    volumes:
      - ./models:/models:ro
      - ./cache:/cache
    command:
      - "--device=gpu:0"
      - "--label_file=/models/labelmap.txt"
      - "--model_file=/models/yolo11n.pt"
      - "--export_imgsz=640"
      - "--export_half"
      - "--export_dynamic"
    gpus: all
```

The `cuda` and `onnx` images both fix runtime to `onnx`. Use `yolo-frigate-cuda` when you want an NVIDIA CUDA base image and use `yolo-frigate-onnx` for the slimmer wheel-based ONNX profile; both use the same NVIDIA + `gpus` pattern for GPU execution.

For **Swarm**, use GPU reservations on `deploy.resources` instead of `gpus: all` (Compose ignores `deploy` on non-Swarm setups).

OpenVINO profile (image name ends with `-openvino`):

```yaml
services:
  yolo-frigate-openvino:
    image: ghcr.io/a-earthperson/yolo-frigate-openvino:v1.2.3
    restart: on-failure
    read_only: true
    devices:
      - /dev/dri:/dev/dri
    group_add:
      - "${GROUP_RENDER}"
    volumes:
      - ./models:/models:ro
      - ./cache:/cache
    command:
      - "--device=gpu"
      - "--label_file=/models/labelmap.txt"
      - "--model_file=/models/yolo11n.pt"
      - "--export_imgsz=320"
      - "--export_half"
      - "--export_dynamic"
```

TFLite / Coral profile (image name ends with `-tflite`):

```yaml
networks:
  yolo-frigate:
    driver: bridge
    internal: true

services:
  yolo-frigate:
    image: ghcr.io/a-earthperson/yolo-frigate-tflite:v1.2.3
    restart: on-failure
    read_only: true
    group_add:
      - "${GROUP_CORAL}"
    security_opt:
      - "no-new-privileges=true"
    networks:
      - yolo-frigate
    volumes:
      - /etc/localtime:/etc/localtime:ro
      - ./models:/models:ro
      - ./cache:/cache
    devices:
      - /dev/apex_0:/dev/apex_0
    command:
      - "--device=pci"
      - "--model_file=/models/yolo11n.pt"
      - "--export_imgsz=320"
```

### Docker Swarm (multi-node, shared storage)

Swarm is useful when **Frigate and workers sit on different nodes** or when **models and export cache** should live on NFS (or another shared filesystem) so any worker can serve the same tree.

Pattern:

- **Overlay network** attached to Frigate and detector services so `http://<service>:8000/detect` resolves cluster-wide.
- **Named volumes** backed by NFS for `/models` and `/cache` so exports done on one node are visible to others (same cache key → same artifact).
- **Placement constraints** so NVIDIA-oriented profiles (`tensorrt`, `cuda`, `onnx`) run only on GPU-labeled nodes and `openvino` on Intel GPU / NPU labeled nodes.
- **Optional** `cap_add: [CAP_PERFMON]` when you want perf counters on Intel stacks; **NVIDIA** stacks often set `NVIDIA_VISIBLE_DEVICES` / `NVIDIA_DRIVER_CAPABILITIES` for full GPU feature exposure inside the container.

Illustrative `stack.yml` (replace NFS address, paths, images, and labels with yours):

```yaml
networks:
  cams:
    external: true

volumes:
  models:
    driver: local
    driver_opts:
      type: nfs
      o: addr=192.168.1.1,rw,nfsvers=4,async
      device: ":/path/on/nfs/to/yolo/models"
  cache:
    driver: local
    driver_opts:
      type: nfs
      o: addr=192.168.1.1,rw,nfsvers=4,async
      device: ":/path/on/nfs/to/yolo/cache"

x-base: &x-base
  cap_add:
    - CAP_PERFMON
  networks:
    - cams
  volumes:
    - /dev/dri:/dev/dri
    - /etc/localtime:/etc/localtime:ro
    - /etc/timezone:/etc/timezone:ro
    - models:/models
    - cache:/cache
    - type: tmpfs
      target: /dev/shm
      tmpfs:
        size: 1073741824

services:
  objectdetector:
    <<: *x-base
    image: ghcr.io/a-earthperson/yolo-frigate-tensorrt:0.1.8
    environment:
      NVIDIA_VISIBLE_DEVICES: all
      NVIDIA_DRIVER_CAPABILITIES: all
    command:
      - "--device=gpu"
      - "--export_imgsz=640"
      - "--export_dynamic"
      - "--label_file=/models/labelmap.txt"
      - "--model_file=/models/yolo26l.pt"
      - "--export_half"
    deploy:
      placement:
        constraints:
          - node.labels.nvidia_gpu_compute == true
      resources:
        limits:
          memory: 4G
        reservations:
          memory: 1200M

  yolov9-intel:
    <<: *x-base
    image: ghcr.io/a-earthperson/yolo-frigate-openvino:0.1.7
    command:
      - "--device=gpu"
      - "--export_imgsz=320"
      - "--export_dynamic"
      - "--label_file=/models/labelmap.txt"
      - "--model_file=/models/yolo26s.pt"
      - "--export_half"
    deploy:
      mode: replicated
      replicas: 0
      placement:
        constraints:
          - node.labels.intel_gpu_compute == true
      resources:
        limits:
          memory: 4096M
        reservations:
          memory: 1280M
```

Keep **one writable cache** per logical deployment; concurrent writers on the same NFS path can corrupt cache entries—usually you run a **single replica** per cache volume or partition cache per replica.

### Kubernetes and other orchestrators

The same container arguments and volume mounts apply: mount models and cache, pass GPU device plugins or `/dev/dri` as your platform requires, and expose port `8000` to clients that call `/detect`.

## Frigate

Frigate’s HTTP detector expects DeepStack-style JSON; point it at this service’s **`/detect`** endpoint on the **Docker network** Frigate shares with the detector (service name resolves under Compose; under Swarm use the stack service name on the overlay).

```yaml
detectors:
  http_detector:
    type: deepstack
    api_url: http://objectdetector:8000/detect
    api_timeout: 1.0
```

Tune `api_timeout` to your SLA: sub-second values work when the model is warm; allow more time for **cold lazy export** or slow hosts. Align Frigate’s `model.labelmap_path` (or Frigate’s label file) with the classes you serve—often the same `labelmap.txt` you pass to `--label_file`.

## Runtime-native scope

The service intentionally does one thing: resolve a runtime profile, lazily export `.pt` sources when needed, load with Ultralytics, and adapt outputs to the existing `Predictions` REST contract. It is **not** a generic server for arbitrary ONNX graphs or custom postprocessing.

## Manual export (optional)

Ultralytics model sources: e.g. [ultralytics/assets](https://github.com/ultralytics/assets).

Lazy export is the default. Prebuilding is useful for air-gapped or reproducible rollouts.

### TensorRT

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest \
  yolo export model=/models/<name>.pt format=engine half=True dynamic=True
```

### OpenVINO

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest \
  yolo export model=/models/<name>.pt format=openvino
```

### ONNX

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest \
  yolo export model=/models/<name>.pt format=onnx
```

### TFLite

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest-cpu \
  yolo export model=/models/<name>.pt format=tflite
```

### EdgeTPU

```bash
docker run -it --rm -v .:/models ultralytics/ultralytics:latest-cpu \
  yolo export model=/models/<name>.pt format=edgetpu
```

You can pass resulting `.engine`, `.onnx`, `*_openvino_model/`, or `.tflite` paths directly to `--model_file`.

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
uv sync --group dev --extra cuda
uv sync --group dev --extra onnx
```

Notes:

- Runtime extras are primarily intended for Linux environments that mirror the Docker images.
- The repo's `tool.uv.sources` pins `torch` / `torchvision` to the CPU or CUDA PyTorch indexes for the `openvino` and `cuda` extras and pins `tensorrt-cu13` to NVIDIA's package index for the `tensorrt` extra.
- `cuda` and `onnx` are separate install profiles that both target the `onnx` runtime backend.

Format and lint:

```bash
uv run ruff check src tests
uv run black src tests
```

The repo pins the default interpreter with [`.python-version`](.python-version); `uv` respects that for `uv sync` and `uv run`.
