from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import cv2
import os
import streamlit as st
from PIL import Image

from config import DEFAULT_ALERT_CONFIDENCE, DEFAULT_CONFIDENCE, DEFAULT_CONFIRM_FRAMES
from detector import (
    ALERTS_DIR,
    SAMPLE_IMAGES,
    decode_image,
    detect,
    load_model,
    save_violation_if_detected,
)

ROOT = Path(__file__).resolve().parent
IS_STREAMLIT_CLOUD = os.path.isdir("/mount/src")


@st.cache_resource
def get_model():
    return load_model()


def render_header():
    st.markdown(
        """
        <div style="padding: 0.5rem 0 1rem 0;">
            <h1 style="margin-bottom: 0.25rem;">Safety & Security Analytics</h1>
            <p style="color: #AAAAAA; margin-top: 0;">
                AI-powered smoking detection using YOLO11
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_metrics(result):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Status", "ALERT" if result.count else "CLEAR")
    c2.metric("Detections", result.count)
    c3.metric("Inference", f"{result.inference_ms:.0f} ms")
    c4.metric("Checked at", datetime.now().strftime("%H:%M:%S"))


def render_detection_table(result):
    if not result.detections:
        st.info("No smoking activity detected in this frame.")
        return

    rows = [
        {
            "#": index,
            "Label": item.label,
            "Confidence": f"{item.confidence * 100:.1f}%",
            "Box (x1, y1, x2, y2)": f"({item.x1}, {item.y1}, {item.x2}, {item.y2})",
        }
        for index, item in enumerate(result.detections, start=1)
    ]
    st.dataframe(rows, use_container_width=True, hide_index=True)


def image_to_download_bytes(image_rgb) -> bytes:
    buffer = BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=95)
    return buffer.getvalue()


def load_sample_image(name: str):
    path = ROOT / name
    if not path.exists():
        st.warning(f"Sample image not found: {name}")
        return None
    return cv2.imread(str(path))


def handle_violation_save(
    result,
    cooldown_seconds: float = 0.0,
    min_alert_confidence: float = DEFAULT_ALERT_CONFIDENCE,
) -> Path | None:
    if "last_violation_saved_at" not in st.session_state:
        st.session_state.last_violation_saved_at = None

    alert_path, saved_at = save_violation_if_detected(
        result,
        ALERTS_DIR,
        st.session_state.last_violation_saved_at,
        cooldown_seconds=cooldown_seconds,
        min_alert_confidence=min_alert_confidence,
    )
    if alert_path:
        st.session_state.last_violation_saved_at = saved_at
    return alert_path


def show_saved_violation_notice(alert_path: Path | None):
    if alert_path:
        st.warning(f"Violation captured and saved to `{alert_path}`")


def render_alerts_sidebar():
    st.markdown("**Violation folder**")
    st.code(str(ALERTS_DIR), language="text")
    if ALERTS_DIR.exists():
        saved = sorted(ALERTS_DIR.glob("violation_*.jpg"), reverse=True)
        st.caption(f"Saved violations: {len(saved)}")
        if saved:
            st.caption(f"Latest: `{saved[0].name}`")
    else:
        st.caption("No violations saved yet.")


def show_detection_result(
    result,
    source_label: str,
    show_download: bool = True,
    alert_path: Path | None = None,
):
    render_metrics(result)
    st.image(result.annotated_rgb, caption="Detection result", use_container_width=True)

    if result.count:
        st.error(f"Smoking activity detected — {result.count} instance(s)")
        show_saved_violation_notice(alert_path)
    else:
        st.success("No smoking activity detected")

    if show_download:
        st.download_button(
            label="Download annotated image",
            data=image_to_download_bytes(result.annotated_rgb),
            file_name=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg",
            mime="image/jpeg",
            use_container_width=True,
        )

    st.markdown("#### Detection details")
    render_detection_table(result)
    st.caption(f"Source: {source_label}")


def stop_local_camera():
    cap = st.session_state.get("local_cap")
    if cap is not None:
        cap.release()
    st.session_state.local_cap = None
    st.session_state.local_monitoring = False


def render_local_live_monitor(
    confidence: float,
    min_alert_confidence: float,
    confirm_frames: int,
):
    st.caption(
        "Automatic live monitoring — frames are analyzed continuously, "
        "like a CCTV feed. Press Stop when finished."
    )

    if "local_monitoring" not in st.session_state:
        st.session_state.local_monitoring = False
    if "local_cap" not in st.session_state:
        st.session_state.local_cap = None

    btn_start, btn_stop = st.columns(2)
    with btn_start:
        if st.button("Start live monitoring", type="primary", use_container_width=True):
            stop_local_camera()
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                st.error("Could not open webcam. Check that a camera is connected.")
            else:
                st.session_state.local_cap = cap
                st.session_state.local_monitoring = True
                st.rerun()
    with btn_stop:
        if st.button("Stop monitoring", use_container_width=True):
            stop_local_camera()
            st.rerun()

    if not st.session_state.local_monitoring or st.session_state.local_cap is None:
        st.info("Press **Start live monitoring** to begin automatic CCTV-style analysis.")
        return

    if "live_consecutive_hits" not in st.session_state:
        st.session_state.live_consecutive_hits = 0

    st.success("Live monitoring active — analyzing automatically…")

    @st.fragment(run_every=timedelta(milliseconds=700))
    def live_monitoring_loop():
        cap = st.session_state.local_cap
        if cap is None or not st.session_state.local_monitoring:
            return

        ret, frame = cap.read()
        if not ret:
            st.warning("Lost camera feed.")
            stop_local_camera()
            return

        result = detect(frame, get_model(), conf=confidence)
        if result.count:
            st.session_state.live_consecutive_hits += 1
        else:
            st.session_state.live_consecutive_hits = 0

        alert_path = None
        if st.session_state.live_consecutive_hits >= confirm_frames:
            alert_path = handle_violation_save(
                result,
                cooldown_seconds=5.0,
                min_alert_confidence=min_alert_confidence,
            )
            if alert_path:
                st.session_state.live_consecutive_hits = 0
        col_feed, col_stats = st.columns([1.2, 1])
        with col_feed:
            st.image(result.annotated_rgb, caption="Live feed", use_container_width=True)
        with col_stats:
            render_metrics(result)
            if result.count:
                st.error(f"Smoking activity detected — {result.count} instance(s)")
                show_saved_violation_notice(alert_path)
            else:
                st.success("Monitoring… no smoking detected")
            render_detection_table(result)

    live_monitoring_loop()


def process_uploaded_video(
    video_bytes: bytes,
    video_name: str,
    confidence: float,
    frame_stride: int,
    cooldown_seconds: float,
    confirm_frames: int,
    min_alert_confidence: float,
):
    temp_dir = ROOT / "temp_uploads"
    temp_dir.mkdir(exist_ok=True)
    temp_path = temp_dir / video_name
    temp_path.write_bytes(video_bytes)

    cap = cv2.VideoCapture(str(temp_path))
    if not cap.isOpened():
        st.error("Could not read the uploaded video.")
        temp_path.unlink(missing_ok=True)
        return None

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    model = get_model()

    progress = st.progress(0, text="Scanning video...")
    status = st.empty()

    frame_idx = 0
    scanned_frames = 0
    detections_found = 0
    saved_captures: list[dict] = []
    last_saved_at = None
    consecutive_hits = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1
        if frame_idx % frame_stride != 0:
            continue

        scanned_frames += 1
        result = detect(frame, model, conf=confidence)

        if result.count:
            consecutive_hits += 1
            detections_found += 1
            if consecutive_hits >= confirm_frames:
                alert_path, last_saved_at = save_violation_if_detected(
                    result,
                    ALERTS_DIR,
                    last_saved_at,
                    cooldown_seconds=cooldown_seconds,
                    min_alert_confidence=min_alert_confidence,
                )
                if alert_path:
                    saved_captures.append(
                        {
                            "frame": frame_idx,
                            "time": frame_idx / fps,
                            "path": alert_path,
                            "result": result,
                        }
                    )
                    consecutive_hits = 0
        else:
            consecutive_hits = 0

        if total_frames > 0:
            progress.progress(
                min(frame_idx / total_frames, 1.0),
                text=f"Scanning frame {frame_idx}/{total_frames}...",
            )
        else:
            status.caption(f"Scanned {scanned_frames} frames...")

    cap.release()
    temp_path.unlink(missing_ok=True)
    progress.empty()
    status.empty()

    return {
        "total_frames": total_frames or frame_idx,
        "scanned_frames": scanned_frames,
        "detections_found": detections_found,
        "saved_captures": saved_captures,
        "fps": fps,
    }


def render_video_results(summary: dict):
    st.markdown("#### Scan summary")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total frames", summary["total_frames"])
    c2.metric("Frames scanned", summary["scanned_frames"])
    c3.metric("Violations detected", summary["detections_found"])
    c4.metric("Images captured", len(summary["saved_captures"]))

    if not summary["saved_captures"]:
        st.success("No smoking activity detected in this video.")
        return

    st.error(f"{len(summary['saved_captures'])} violation image(s) captured and saved.")
    st.markdown("#### Captured violation images")

    for index, item in enumerate(summary["saved_captures"], start=1):
        minutes = int(item["time"] // 60)
        seconds = int(item["time"] % 60)
        st.markdown(
            f"**Capture {index}** — frame {item['frame']} "
            f"({minutes:02d}:{seconds:02d})"
        )
        st.image(
            item["result"].annotated_rgb,
            caption=item["path"].name,
            use_container_width=True,
        )
        st.download_button(
            label=f"Download capture {index}",
            data=image_to_download_bytes(item["result"].annotated_rgb),
            file_name=item["path"].name,
            mime="image/jpeg",
            key=f"video_capture_{index}_{item['frame']}",
        )


def render_video_analysis(
    confidence: float,
    frame_stride: int,
    cooldown_seconds: float,
    confirm_frames: int,
    min_alert_confidence: float,
):
    st.caption(
        "Upload a video to scan frame-by-frame. Violations are saved only when "
        "detection is strong and confirmed across multiple frames."
    )
    st.warning(
        "This demo model was not trained on your CCTV footage. "
        "Low-confidence alerts on tables, shadows, and objects are expected. "
        "Use confidence 0.55+ and confirm frames to reduce false alarms."
    )

    uploaded = st.file_uploader(
        "Upload a video",
        type=["mp4", "avi", "mov", "mkv", "webm"],
    )

    if uploaded is None:
        st.info("Upload a video file to start analysis.")
        return

    st.video(uploaded)
    analyze = st.button("Analyze video", type="primary", use_container_width=True)

    if not analyze:
        return

    with st.spinner("Processing video. This may take a minute for longer files..."):
        summary = process_uploaded_video(
            uploaded.read(),
            uploaded.name,
            confidence,
            frame_stride,
            cooldown_seconds,
            confirm_frames,
            min_alert_confidence,
        )

    if summary:
        render_video_results(summary)


def render_cloud_camera_monitor(confidence: float):
    st.caption(
        "Cloud demo mode: browser security requires a manual capture on the free Streamlit link. "
        "For automatic live CCTV, run the app locally (see sidebar)."
    )

    col_cam, col_result = st.columns(2, gap="large")

    with col_cam:
        camera = st.camera_input("Allow camera, then tap capture to scan")

    with col_result:
        st.subheader("Scan result")
        if camera is None:
            st.info("Capture a frame from the camera to analyze it.")
        else:
            frame_bytes = camera.getvalue()
            frame_hash = hash(frame_bytes)
            if st.session_state.get("camera_frame_hash") != frame_hash:
                input_image = decode_image(frame_bytes)
                st.session_state.camera_frame_hash = frame_hash
                st.session_state.camera_result = detect(
                    input_image,
                    get_model(),
                    conf=confidence,
                )
                st.session_state.camera_alert_path = handle_violation_save(
                    st.session_state.camera_result,
                    min_alert_confidence=DEFAULT_ALERT_CONFIDENCE,
                )

            show_detection_result(
                st.session_state.camera_result,
                "Camera capture",
                alert_path=st.session_state.get("camera_alert_path"),
            )

    st.warning(
        "The public Streamlit Cloud link cannot access your webcam continuously in the background. "
        "Run this command on your computer for automatic live monitoring:\n\n"
        "`py -m streamlit run streamlit_app.py`"
    )


st.set_page_config(
    page_title="Safety & Security Analytics",
    page_icon="🚭",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_header()

with st.spinner("Loading AI model..."):
    get_model()

with st.sidebar:
    st.header("Settings")
    confidence = st.slider(
        "Confidence threshold",
        0.40,
        0.90,
        DEFAULT_CONFIDENCE,
        0.05,
        help="Ignore detections below this score. 0.55+ reduces false alarms.",
    )
    min_alert_confidence = st.slider(
        "Minimum confidence to save violation",
        0.40,
        0.90,
        DEFAULT_ALERT_CONFIDENCE,
        0.05,
        help="Only capture images when the top detection is at least this confident.",
    )
    confirm_frames = st.slider(
        "Confirm frames (video/live)",
        1,
        5,
        DEFAULT_CONFIRM_FRAMES,
        1,
        help="Require detections in this many consecutive scanned frames before saving.",
    )
    frame_stride = st.slider("Video frame skip", 1, 30, 5, 1)
    st.caption("Analyze every Nth frame in uploaded videos (higher = faster).")
    capture_cooldown = st.slider("Capture cooldown (seconds)", 1, 30, 3, 1)
    st.caption("Minimum gap between saved violation images in video/live mode.")
    st.divider()
    st.markdown("**Automatic live CCTV**")
    st.code("py -m streamlit run streamlit_app.py", language="bash")
    st.caption(
        "Run on your PC, open the local URL, then use Camera Monitor → Start live monitoring."
    )
    st.divider()
    st.markdown("**CLI live mode**")
    st.code("py app.py", language="bash")
    st.divider()
    if IS_STREAMLIT_CLOUD:
        st.info("You are on Streamlit Cloud (manual camera capture only).")
    else:
        st.success("Local mode — automatic live monitoring is available.")
    st.divider()
    render_alerts_sidebar()
    st.divider()
    st.caption("Proof-of-concept demo. Alerts, CCTV integration, and dashboards can be added in the next phase.")

tab_live, tab_image, tab_video, tab_about = st.tabs(
    ["Camera Monitor", "Image Analysis", "Video Analysis", "About"]
)

with tab_live:
    st.subheader("Camera monitoring")
    if IS_STREAMLIT_CLOUD:
        render_cloud_camera_monitor(confidence)
    else:
        render_local_live_monitor(confidence, min_alert_confidence, confirm_frames)

with tab_image:
    col_input, col_output = st.columns(2, gap="large")

    with col_input:
        st.subheader("Input")
        source = st.radio(
            "Input source",
            ["Sample image", "Upload image"],
            horizontal=True,
        )

        input_image = None
        source_label = "No input selected"

        if source == "Sample image":
            available = [name for name in SAMPLE_IMAGES if (ROOT / name).exists()]
            if available:
                selected = st.selectbox("Choose a sample", available)
                input_image = load_sample_image(selected)
                source_label = selected
                if input_image is not None:
                    st.image(
                        cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
                        caption=f"Sample: {selected}",
                        use_container_width=True,
                    )
            else:
                st.warning("No sample images found in the repository.")
        else:
            uploaded = st.file_uploader(
                "Upload an image",
                type=["jpg", "jpeg", "png", "webp"],
            )
            if uploaded:
                input_image = decode_image(uploaded.read())
                source_label = uploaded.name
                st.image(
                    cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
                    caption=uploaded.name,
                    use_container_width=True,
                )

    with col_output:
        st.subheader("Result")

        if input_image is None:
            st.info("Select a sample or upload an image to run detection.")
        else:
            with st.spinner("Analyzing frame..."):
                result = detect(input_image, get_model(), conf=confidence)
                alert_path = handle_violation_save(result)

            show_detection_result(result, source_label, alert_path=alert_path)

with tab_video:
    st.subheader("Video analysis")
    render_video_analysis(
        confidence,
        frame_stride,
        capture_cooldown,
        confirm_frames,
        min_alert_confidence,
    )

with tab_about:
    st.subheader("About this demo")
    st.markdown(
        """
        This module is part of a **Safety & Security Analytics** platform. It uses a
        YOLO11 model to identify smoking activity in live camera feeds and images.

        **Video analysis**
        - Upload MP4/AVI/MOV and scan frame-by-frame
        - Violation frames are auto-captured to `alerts/`
        - Download captured evidence from the results panel

        **Automatic violation capture**
        - Detected violations are saved to the `alerts/` folder
        - Each image includes bounding boxes and confidence score
        - Live monitoring saves a new snapshot every 5 seconds while violation continues

        **Automatic live monitoring (local)**
        - Run `py -m streamlit run streamlit_app.py` on your computer
        - Open **Camera Monitor** → **Start live monitoring**
        - Frames are analyzed continuously like CCTV

        **Cloud link (Streamlit Cloud)**
        - Manual camera capture per scan
        - Image upload and sample testing
        - Best for sharing a quick demo link

        **Planned next phase**
        - Email / SMS / WhatsApp alerts
        - Automatic incident snapshots
        - RTSP CCTV stream integration
        - Admin dashboard and audit logs
        """
    )
