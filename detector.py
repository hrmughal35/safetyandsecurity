from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "best.onnx"
SAMPLE_IMAGES = ["output2.jpg", "output1.jpg", "output.jpg"]
LABEL_FIXES = {"Smooking": "Smoking", "smooking": "Smoking"}
_MODEL_CACHE: YOLO | None = None


def normalize_label(label: str) -> str:
    return LABEL_FIXES.get(label, label)


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
    global _MODEL_CACHE
    if _MODEL_CACHE is None:
        _MODEL_CACHE = YOLO(str(model_path))
    return _MODEL_CACHE


def decode_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image.")
    return image


def annotate_frame(image_bgr: np.ndarray, detections: list[Detection]) -> np.ndarray:
    annotated = image_bgr.copy()
    color = (255, 89, 71)

    for item in detections:
        cv2.rectangle(annotated, (item.x1, item.y1), (item.x2, item.y2), color, 2)
        label = f"{item.label} {item.confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.6
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(label, font, scale, thickness)
        top = max(item.y1, text_height + 12)
        cv2.rectangle(
            annotated,
            (item.x1, top - text_height - 8),
            (item.x1 + text_width + 8, top),
            color,
            -1,
        )
        cv2.putText(
            annotated,
            label,
            (item.x1 + 4, top - 4),
            font,
            scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return annotated


def detect(image_bgr: np.ndarray, model: YOLO, conf: float = 0.25) -> DetectionResult:
    start = time.perf_counter()
    results = model(image_bgr, conf=conf, verbose=False)
    result = results[0]
    inference_ms = (time.perf_counter() - start) * 1000

    detections: list[Detection] = []
    if result.boxes is not None:
        names = result.names
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls = int(box.cls[0])
            detections.append(
                Detection(
                    label=normalize_label(str(names[cls])),
                    confidence=float(box.conf[0]),
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                )
            )

    annotated = annotate_frame(image_bgr, detections)
    return DetectionResult(annotated, detections, inference_ms)


ALERTS_DIR = Path(__file__).resolve().parent / "alerts"


def save_violation_alert(result: DetectionResult, alerts_dir: Path = ALERTS_DIR) -> Path:
    alerts_dir.mkdir(parents=True, exist_ok=True)
    top_confidence = max((item.confidence for item in result.detections), default=0.0)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    filename = f"violation_{timestamp}_{ms:03d}_conf{top_confidence:.2f}.jpg"
    path = alerts_dir / filename
    cv2.imwrite(str(path), result.annotated_bgr)
    return path


def save_violation_if_detected(
    result: DetectionResult,
    alerts_dir: Path = ALERTS_DIR,
    last_saved_at: float | None = None,
    cooldown_seconds: float = 0.0,
) -> tuple[Path | None, float | None]:
    if result.count == 0:
        return None, last_saved_at

    now = time.time()
    if (
        last_saved_at is not None
        and cooldown_seconds > 0
        and (now - last_saved_at) < cooldown_seconds
    ):
        return None, last_saved_at

    path = save_violation_alert(result, alerts_dir)
    return path, now


def save_alert(image_bgr: np.ndarray, alerts_dir: Path) -> Path:
    alerts_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ms = int((time.time() % 1) * 1000)
    path = alerts_dir / f"violation_{timestamp}_{ms:03d}.jpg"
    cv2.imwrite(str(path), image_bgr)
    return path
