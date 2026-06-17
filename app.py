import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

from detector import MODEL_PATH, detect, save_alert


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def print_summary(result):
    print(f"Detections: {result.count}")
    print(f"Inference: {result.inference_ms:.1f} ms")
    for index, item in enumerate(result.detections, start=1):
        print(
            f"  [{index}] {item.label} "
            f"({item.confidence * 100:.1f}%) "
            f"box=({item.x1},{item.y1},{item.x2},{item.y2})"
        )


def run_image(model, source: str, output: Path | None, conf: float, alerts_dir: Path | None):
    image = cv2.imread(source)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {source}")

    result = detect(image, model, conf=conf)
    print_summary(result)

    if output:
        cv2.imwrite(str(output), result.annotated_bgr)
        print(f"Saved result to {output}")

    if result.count and alerts_dir:
        alert_path = save_alert(result.annotated_bgr, alerts_dir)
        print(f"Alert saved to {alert_path}")
    elif not output:
        cv2.imshow("YOLO Inference", result.annotated_bgr)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return result.count


def run_video(model, source, output: Path | None, conf: float, alerts_dir: Path | None):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    writer = None
    if output:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS) or 20
        writer = cv2.VideoWriter(
            str(output),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

    alert_saved = False
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = detect(frame, model, conf=conf)
            if result.count and alerts_dir and not alert_saved:
                alert_path = save_alert(result.annotated_bgr, alerts_dir)
                print(f"Alert saved to {alert_path}")
                alert_saved = True

            if writer:
                writer.write(result.annotated_bgr)
            else:
                cv2.imshow("YOLO Inference", result.annotated_bgr)
                if cv2.waitKey(1) == 27:
                    break
    finally:
        cap.release()
        if writer:
            writer.release()
            print(f"Saved result to {output}")
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Smoking detection with YOLO11")
    parser.add_argument(
        "--source",
        default="0",
        help="Webcam index (0), image path, or video path",
    )
    parser.add_argument(
        "--output",
        help="Optional output path for image (.jpg) or video (.mp4)",
    )
    parser.add_argument(
        "--model",
        default=str(MODEL_PATH),
        help="Path to ONNX model",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold (0-1)",
    )
    parser.add_argument(
        "--alerts-dir",
        default="alerts",
        help="Folder to save alert snapshots when smoking is detected",
    )
    args = parser.parse_args()

    source = parse_source(args.source)
    output = Path(args.output) if args.output else None
    alerts_dir = Path(args.alerts_dir) if args.alerts_dir else None
    model = YOLO(args.model)

    if isinstance(source, int) or str(source).lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        run_video(model, source, output, args.conf, alerts_dir)
    else:
        if not Path(source).exists():
            raise FileNotFoundError(f"Source not found: {source}")
        run_image(model, source, output, args.conf, alerts_dir)


if __name__ == "__main__":
    main()
