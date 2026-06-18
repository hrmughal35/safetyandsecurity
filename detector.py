from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

MODEL_PATH = Path(__file__).resolve().parent / "best.onnx"
SAMPLE_IMAGES = ["output2.jpg", "output1.jpg", "output.jpg"]
LABEL_FIXES = {"Smooking": "Smoking", "smooking": "Smoking"}
CLASS_NAMES = {0: "Smoking"}
INPUT_SIZE = 640
_MODEL_CACHE = None


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


class OnnxDetector:
    def __init__(self, model_path: Path | str = MODEL_PATH):
        import onnxruntime as ort

        self.session = ort.InferenceSession(
            str(model_path),
            providers=["CPUExecutionProvider"],
        )
        self.input_name = self.session.get_inputs()[0].name


def load_model(model_path: Path | str = MODEL_PATH):
    global _MODEL_CACHE
    model_path = Path(model_path)

    if _MODEL_CACHE is not None and model_path == MODEL_PATH:
        return _MODEL_CACHE

    try:
        from ultralytics import YOLO

        loaded = ("ultralytics", YOLO(str(model_path)))
    except ImportError:
        loaded = ("onnx", OnnxDetector(model_path))

    if model_path == MODEL_PATH:
        _MODEL_CACHE = loaded
    return loaded


def decode_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image.")
    return image


def letterbox(image_bgr: np.ndarray, size: int = INPUT_SIZE):
    height, width = image_bgr.shape[:2]
    scale = min(size / height, size / width)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image_bgr, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_w = size - new_width
    pad_h = size - new_height
    left = pad_w // 2
    right = pad_w - left
    top = pad_h // 2
    bottom = pad_h - top
    padded = cv2.copyMakeBorder(
        resized,
        top,
        bottom,
        left,
        right,
        cv2.BORDER_CONSTANT,
        value=(114, 114, 114),
    )
    rgb = padded[:, :, ::-1]
    blob = rgb.transpose(2, 0, 1)[None].astype(np.float32) / 255.0
    return blob, scale, left, top


def nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float = 0.45) -> list[int]:
    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[current], x1[rest])
        yy1 = np.maximum(y1[current], y1[rest])
        xx2 = np.minimum(x2[current], x2[rest])
        yy2 = np.minimum(y2[current], y2[rest])

        widths = np.maximum(0.0, xx2 - xx1)
        heights = np.maximum(0.0, yy2 - yy1)
        intersection = widths * heights
        union = areas[current] + areas[rest] - intersection
        iou = intersection / np.maximum(union, 1e-6)
        order = rest[iou <= iou_threshold]

    return keep


def detect_onnx(image_bgr: np.ndarray, model: OnnxDetector, conf: float = 0.25) -> list[Detection]:
    blob, scale, pad_x, pad_y = letterbox(image_bgr)
    outputs = model.session.run(None, {model.input_name: blob})[0]
    predictions = np.squeeze(outputs).T

    boxes: list[list[float]] = []
    scores: list[float] = []
    classes: list[int] = []

    for row in predictions:
        class_scores = row[4:]
        class_id = int(np.argmax(class_scores))
        score = float(class_scores[class_id])
        if score < conf:
            continue

        cx, cy, width, height = row[:4]
        x1 = (cx - width / 2 - pad_x) / scale
        y1 = (cy - height / 2 - pad_y) / scale
        x2 = (cx + width / 2 - pad_x) / scale
        y2 = (cy + height / 2 - pad_y) / scale

        image_h, image_w = image_bgr.shape[:2]
        x1 = float(np.clip(x1, 0, image_w - 1))
        y1 = float(np.clip(y1, 0, image_h - 1))
        x2 = float(np.clip(x2, 0, image_w - 1))
        y2 = float(np.clip(y2, 0, image_h - 1))

        boxes.append([x1, y1, x2, y2])
        scores.append(score)
        classes.append(class_id)

    if not boxes:
        return []

    boxes_np = np.array(boxes, dtype=np.float32)
    scores_np = np.array(scores, dtype=np.float32)
    keep = nms(boxes_np, scores_np)

    detections: list[Detection] = []
    for index in keep:
        label = normalize_label(CLASS_NAMES.get(classes[index], f"class_{classes[index]}"))
        x1, y1, x2, y2 = boxes_np[index]
        detections.append(
            Detection(
                label=label,
                confidence=scores_np[index],
                x1=int(x1),
                y1=int(y1),
                x2=int(x2),
                y2=int(y2),
            )
        )
    return detections


def detect_ultralytics(image_bgr: np.ndarray, model, conf: float = 0.25) -> list[Detection]:
    results = model(image_bgr, conf=conf, verbose=False)
    result = results[0]
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
    return detections


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


def detect(image_bgr: np.ndarray, model=None, conf: float = 0.25) -> DetectionResult:
    if model is None:
        backend, engine = load_model()
    elif isinstance(model, tuple):
        backend, engine = model
    else:
        backend, engine = "ultralytics", model

    start = time.perf_counter()

    if backend == "onnx":
        detections = detect_onnx(image_bgr, engine, conf=conf)
    else:
        detections = detect_ultralytics(image_bgr, engine, conf=conf)

    inference_ms = (time.perf_counter() - start) * 1000
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
