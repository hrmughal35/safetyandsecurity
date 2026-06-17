from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path

import cv2
import os
import streamlit as st
from PIL import Image

from detector import SAMPLE_IMAGES, decode_image, detect, load_model

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


def show_detection_result(result, source_label: str, show_download: bool = True):
    render_metrics(result)
    st.image(result.annotated_rgb, caption="Detection result", use_container_width=True)

    if result.count:
        st.error(f"Smoking activity detected — {result.count} instance(s)")
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


def render_local_live_monitor(confidence: float):
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
        col_feed, col_stats = st.columns([1.2, 1])
        with col_feed:
            st.image(result.annotated_rgb, caption="Live feed", use_container_width=True)
        with col_stats:
            render_metrics(result)
            if result.count:
                st.error(f"Smoking activity detected — {result.count} instance(s)")
            else:
                st.success("Monitoring… no smoking detected")
            render_detection_table(result)

    live_monitoring_loop()


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

            show_detection_result(
                st.session_state.camera_result,
                "Camera capture",
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
    confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
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
    st.caption("Proof-of-concept demo. Alerts, CCTV integration, and dashboards can be added in the next phase.")

tab_live, tab_image, tab_about = st.tabs(["Camera Monitor", "Image Analysis", "About"])

with tab_live:
    st.subheader("Camera monitoring")
    if IS_STREAMLIT_CLOUD:
        render_cloud_camera_monitor(confidence)
    else:
        render_local_live_monitor(confidence)

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

            show_detection_result(result, source_label)

with tab_about:
    st.subheader("About this demo")
    st.markdown(
        """
        This module is part of a **Safety & Security Analytics** platform. It uses a
        YOLO11 model to identify smoking activity in live camera feeds and images.

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
