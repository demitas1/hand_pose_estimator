import { HandLandmarker, FilesetResolver, DrawingUtils } from "@mediapipe/tasks-vision";

class HandTracker {
  private handLandmarker: HandLandmarker | null = null;
  private drawingUtils: DrawingUtils | null = null;
  private video: HTMLVideoElement | null = null;
  private canvas: HTMLCanvasElement | null = null;
  private ctx: CanvasRenderingContext2D | null = null;

  constructor() {
    this.initializeDOMElements();
    this.initializeMediaPipe();
  }

  private initializeDOMElements(): void {
    this.video = document.getElementById("video") as HTMLVideoElement;
    this.canvas = document.getElementById("canvas") as HTMLCanvasElement;
    this.ctx = this.canvas.getContext("2d");
    this.drawingUtils = new DrawingUtils(this.ctx!);
  }

  private async initializeMediaPipe(): Promise<void> {
    const vision = await FilesetResolver.forVisionTasks(
      "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@latest/wasm"
    );
    this.handLandmarker = await HandLandmarker.createFromOptions(vision, {
      baseOptions: {
        modelAssetPath: "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task",
        delegate: "GPU"
      },
      runningMode: "VIDEO",
      numHands: 2
    });
    this.startCamera();
  }

  private async startCamera(): Promise<void> {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (this.video) {
        this.video.srcObject = stream;
        this.video.addEventListener("loadeddata", () => this.predictWebcam());
      }
    } catch (err) {
      console.error("Error accessing the camera:", err);
    }
  }

  private async predictWebcam(): Promise<void> {
    let startTimeMs = performance.now();
    if (this.handLandmarker && this.video) {
      const results = this.handLandmarker.detectForVideo(this.video, startTimeMs);
      this.displayResults(results);
    }
    requestAnimationFrame(() => this.predictWebcam());
  }

  private displayResults(results: any): void {
    if (this.ctx && this.canvas) {
      this.ctx.save();
      this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
      if (results.landmarks) {
        for (const landmarks of results.landmarks) {
          this.drawingUtils!.drawConnectors(
            landmarks,
            HandLandmarker.HAND_CONNECTIONS,
            { color: "#00FF00", lineWidth: 5 }
          );
          this.drawingUtils!.drawLandmarks(landmarks, {
            color: "#FF0000",
            lineWidth: 2
          });
        }
      }
      this.ctx.restore();
    }
  }
}

// Initialize the hand tracker when the page loads
window.onload = () => new HandTracker();
