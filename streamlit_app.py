from datetime import datetime
from io import BytesIO
from pathlib import Path

import cv2
import streamlit as st
from PIL import Image

from detector import SAMPLE_IMAGES, decode_image, detect, load_model

ROOT = Path(__file__).resolve().parent


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


def show_detection_result(result, source_label: str):
    render_metrics(result)
    st.image(result.annotated_rgb, caption="Detection result", use_container_width=True)

    if result.count:
        st.error(f"Smoking activity detected — {result.count} instance(s)")
        st.toast("Alert: smoking detected", icon="🚨")
    else:
        st.success("No smoking activity detected")

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
    st.markdown("**How to demo**")
    st.markdown(
        "1. Open **Camera Monitor**\n"
        "2. Allow camera access\n"
        "3. Tap capture to scan a frame\n"
        "4. Or use **Image Analysis** for samples/upload"
    )
    st.divider()
    st.markdown("**True live CCTV on your PC**")
    st.code("py app.py", language="bash")
    st.caption("Continuous webcam monitoring works locally, not on free cloud hosting.")
    st.divider()
    st.caption("Proof-of-concept demo. Alerts, CCTV integration, and dashboards can be added in the next phase.")

tab_live, tab_image, tab_about = st.tabs(["Camera Monitor", "Image Analysis", "About"])

with tab_live:
    st.subheader("Camera monitoring")
    st.caption(
        "Point your camera at the scene and tap the camera button to scan. "
        "Each capture is analyzed instantly — like a CCTV snapshot check."
    )

    col_cam, col_result = st.columns(2, gap="large")

    with col_cam:
        camera = st.camera_input("Allow camera, then tap capture to scan")
        if camera:
            st.success("Frame captured — analyzing below.")

    with col_result:
        st.subheader("Scan result")
        if camera is None:
            st.info("Capture a frame from the camera to start monitoring.")
        else:
            frame_bytes = camera.getvalue()
            frame_hash = hash(frame_bytes)
            if st.session_state.get("camera_frame_hash") != frame_hash:
                with st.spinner("Analyzing frame..."):
                    input_image = decode_image(frame_bytes)
                    st.session_state.camera_frame_hash = frame_hash
                    st.session_state.camera_result = detect(
                        input_image,
                        get_model(),
                        conf=confidence,
                    )
                    st.session_state.camera_source = "Camera capture"

            show_detection_result(
                st.session_state.camera_result,
                st.session_state.get("camera_source", "Camera capture"),
            )

    st.info(
        "On free cloud hosting, browser live video streaming is limited. "
        "For uninterrupted real-time CCTV-style monitoring, run `py app.py` on your computer."
    )

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
        YOLO11 model to identify smoking activity in camera captures and images.

        **Web demo (Streamlit Cloud)**
        - Camera capture with instant scan
        - Image upload and sample testing
        - Bounding boxes with confidence scores
        - Downloadable evidence image

        **Local demo (full live CCTV)**
        - Run `py app.py` on your computer
        - Continuous webcam monitoring with live bounding boxes

        **Planned next phase**
        - Email / SMS / WhatsApp alerts
        - Automatic incident snapshots
        - RTSP CCTV stream integration
        - Admin dashboard and audit logs
        """
    )
