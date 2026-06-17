from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "best.onnx"
SAMPLE_IMAGES = ["output2.jpg", "output1.jpg", "output.jpg"]


@dataclass
class Detection:
    label: str
    confidence: float
    x1: int
    y1: int
    x2: int
    y2: int


@dataclass
class DetectionResult:
    annotated_bgr: np.ndarray
    detections: list[Detection]
    inference_ms: float

    @property
    def count(self) -> int:
        return len(self.detections)

    @property
    def annotated_rgb(self) -> np.ndarray:
        return cv2.cvtColor(self.annotated_bgr, cv2.COLOR_BGR2RGB)


def load_model(model_path: Path | str = MODEL_PATH) -> YOLO:
    return YOLO(str(model_path))


def decode_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image.")
    return image


def detect(image_bgr: np.ndarray, model: YOLO, conf: float = 0.25) -> DetectionResult:
    start = time.perf_counter()
    results = model(image_bgr, conf=conf, verbose=False)
    result = results[0]
    annotated = result.plot()
    inference_ms = (time.perf_counter() - start) * 1000

    detections: list[Detection] = []
    if result.boxes is not None:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            detections.append(
                Detection(
                    label=result.names[cls],
                    confidence=float(box.conf[0]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    return DetectionResult(annotated, detections, inference_ms)


def save_alert(image_bgr: np.ndarray, alerts_dir: Path) -> Path:
    alerts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = alerts_dir / f"smoking_alert_{timestamp}.jpg"
    cv2.imwrite(str(path), image_bgr)
    return path
