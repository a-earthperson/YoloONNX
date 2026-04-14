const DETECT_ENDPOINT = "/detect";
const DEFAULT_DETECTION_FPS = 2;
const JPEG_QUALITY = 0.85;

class LiveDetectorApp {
  constructor() {
    this.video = document.getElementById("source-video");
    this.canvas = document.getElementById("preview-canvas");
    this.statusText = document.getElementById("status-text");
    this.metricsText = document.getElementById("metrics-text");
    this.fpsInput = document.getElementById("fps-input");

    this.context = this.canvas.getContext("2d");
    this.captureCanvas = document.createElement("canvas");
    this.captureContext = this.captureCanvas.getContext("2d");

    this.predictions = [];
    this.detectionTimer = null;
    this.renderHandle = null;
    this.inFlight = false;
    this.loopVersion = 0;
    this.requestSequence = 0;
    this.lastAppliedSequence = 0;
    this.lastDetectionLatencyMs = null;
    this.lastDetectionAt = null;
  }

  async start() {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error("This browser does not support camera capture.");
    }
    if (!this.context || !this.captureContext) {
      throw new Error("Canvas 2D rendering is unavailable.");
    }

    const fps = this.parseFps(this.fpsInput.value);
    this.fpsInput.value = String(fps);
    this.fpsInput.addEventListener("change", () => this.restartDetectionLoop());
    this.fpsInput.addEventListener("input", () => this.restartDetectionLoop());

    this.statusText.textContent = "Requesting camera permission...";

    const stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: {
        facingMode: "environment",
      },
    });

    this.video.srcObject = stream;
    await this.video.play();
    await this.waitForVideoMetadata();

    this.resizeCanvases(this.video.videoWidth, this.video.videoHeight);
    this.statusText.textContent = "Camera ready. Running detections.";
    this.metricsText.textContent = "Waiting for first detection...";

    this.startRenderLoop();
    this.restartDetectionLoop();
  }

  parseFps(rawValue) {
    const parsed = Number.parseFloat(rawValue);
    if (!Number.isFinite(parsed) || parsed <= 0) {
      return DEFAULT_DETECTION_FPS;
    }
    return Math.min(parsed, 30);
  }

  waitForVideoMetadata() {
    if (this.video.readyState >= HTMLMediaElement.HAVE_METADATA) {
      return Promise.resolve();
    }

    return new Promise((resolve) => {
      this.video.addEventListener("loadedmetadata", () => resolve(), {
        once: true,
      });
    });
  }

  resizeCanvases(width, height) {
    this.canvas.width = width;
    this.canvas.height = height;
    this.captureCanvas.width = width;
    this.captureCanvas.height = height;
  }

  startRenderLoop() {
    const renderFrame = () => {
      if (this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        this.context.drawImage(this.video, 0, 0, this.canvas.width, this.canvas.height);
        this.drawPredictions();
      }
      this.renderHandle = window.requestAnimationFrame(renderFrame);
    };

    renderFrame();
  }

  restartDetectionLoop() {
    if (this.detectionTimer !== null) {
      window.clearTimeout(this.detectionTimer);
    }
    this.loopVersion += 1;
    const activeLoopVersion = this.loopVersion;
    const intervalMs = 1000 / this.parseFps(this.fpsInput.value);
    this.detectionTimer = window.setTimeout(() => {
      void this.runDetectionLoop(intervalMs, activeLoopVersion);
    }, intervalMs);
  }

  async runDetectionLoop(intervalMs, loopVersion) {
    try {
      if (!this.inFlight && this.video.readyState >= HTMLMediaElement.HAVE_CURRENT_DATA) {
        await this.detectCurrentFrame();
      }
    } finally {
      if (loopVersion !== this.loopVersion) {
        return;
      }
      this.detectionTimer = window.setTimeout(() => {
        void this.runDetectionLoop(intervalMs, loopVersion);
      }, intervalMs);
    }
  }

  async detectCurrentFrame() {
    this.inFlight = true;
    const requestId = ++this.requestSequence;
    const startedAt = performance.now();

    try {
      const frame = await this.captureFrame();
      const result = await this.requestDetections(frame);
      if (requestId >= this.lastAppliedSequence) {
        this.predictions = Array.isArray(result.predictions) ? result.predictions : [];
        this.lastAppliedSequence = requestId;
        this.lastDetectionLatencyMs = performance.now() - startedAt;
        this.lastDetectionAt = new Date();
        this.statusText.textContent = "Camera ready. Running detections.";
        this.metricsText.textContent = this.describeMetrics();
      }
    } catch (error) {
      this.statusText.textContent = `Detection failed: ${this.formatError(error)}`;
    } finally {
      this.inFlight = false;
    }
  }

  captureFrame() {
    this.captureContext.drawImage(
      this.video,
      0,
      0,
      this.captureCanvas.width,
      this.captureCanvas.height,
    );

    return new Promise((resolve, reject) => {
      this.captureCanvas.toBlob(
        (blob) => {
          if (blob) {
            resolve(blob);
            return;
          }
          reject(new Error("Failed to encode the current frame."));
        },
        "image/jpeg",
        JPEG_QUALITY,
      );
    });
  }

  async requestDetections(frameBlob) {
    const formData = new FormData();
    formData.append("image", frameBlob, "frame.jpg");

    const response = await fetch(DETECT_ENDPOINT, {
      method: "POST",
      body: formData,
      cache: "no-store",
    });

    if (!response.ok) {
      let message = `HTTP ${response.status}`;
      try {
        const payload = await response.json();
        if (payload?.detail) {
          message = payload.detail;
        }
      } catch {
        // Leave the fallback status text intact when the error body is not JSON.
      }
      throw new Error(message);
    }

    return response.json();
  }

  drawPredictions() {
    this.context.lineWidth = 3;
    this.context.font = "16px Arial, sans-serif";
    this.context.textBaseline = "top";

    for (const prediction of this.predictions) {
      const x = prediction.x_min;
      const y = prediction.y_min;
      const width = prediction.x_max - prediction.x_min;
      const height = prediction.y_max - prediction.y_min;

      this.context.strokeStyle = "#22c55e";
      this.context.strokeRect(x, y, width, height);

      const label = `${prediction.label} ${(prediction.confidence * 100).toFixed(1)}%`;
      const textWidth = this.context.measureText(label).width;
      const labelX = x;
      const labelY = Math.max(0, y - 24);

      this.context.fillStyle = "#22c55e";
      this.context.fillRect(labelX, labelY, textWidth + 12, 22);
      this.context.fillStyle = "#020617";
      this.context.fillText(label, labelX + 6, labelY + 3);
    }
  }

  describeMetrics() {
    const latency = this.lastDetectionLatencyMs === null
      ? "n/a"
      : `${this.lastDetectionLatencyMs.toFixed(0)} ms`;
    const timestamp = this.lastDetectionAt === null
      ? "never"
      : this.lastDetectionAt.toLocaleTimeString();
    return `${this.predictions.length} detections, last update ${timestamp}, latency ${latency}`;
  }

  formatError(error) {
    if (error instanceof Error) {
      return error.message;
    }
    return String(error);
  }
}

async function main() {
  const app = new LiveDetectorApp();
  try {
    await app.start();
  } catch (error) {
    app.statusText.textContent = `Unable to start live view: ${app.formatError(error)}`;
    app.metricsText.textContent = "Check browser permissions and server availability.";
  }
}

void main();
