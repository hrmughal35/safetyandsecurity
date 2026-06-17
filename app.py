import argparse
from pathlib import Path

import cv2
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "best.onnx"


def parse_source(source: str):
    if source.isdigit():
        return int(source)
    return source


def run_image(model, source: str, output: Path | None):
    results = model(source)
    annotated = results[0].plot()
    count = len(results[0].boxes) if results[0].boxes is not None else 0

    if output:
        cv2.imwrite(str(output), annotated)
        print(f"Saved result to {output}")
    else:
        cv2.imshow("YOLO Inference", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    print(f"Detections: {count}")
    return count


def run_video(model, source, output: Path | None):
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

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            result = model(frame)[0]
            annotated = result.plot()

            if writer:
                writer.write(annotated)
            else:
                cv2.imshow("YOLO Inference", annotated)
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
    args = parser.parse_args()

    source = parse_source(args.source)
    output = Path(args.output) if args.output else None
    model = YOLO(args.model)

    if isinstance(source, int) or str(source).lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        run_video(model, source, output)
    else:
        if not Path(source).exists():
            raise FileNotFoundError(f"Source not found: {source}")
        run_image(model, source, output)


if __name__ == "__main__":
    main()
