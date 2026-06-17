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


st.set_page_config(
    page_title="Safety & Security Analytics",
    page_icon="🚭",
    layout="wide",
    initial_sidebar_state="expanded",
)

render_header()

with st.sidebar:
    st.header("Settings")
    confidence = st.slider("Confidence threshold", 0.1, 0.9, 0.25, 0.05)
    st.divider()
    st.markdown("**How to demo**")
    st.markdown(
        "1. Try a sample image below\n"
        "2. Or upload your own photo\n"
        "3. Or use camera capture\n"
        "4. Review alert status and download proof"
    )
    st.divider()
    st.caption("Proof-of-concept demo. Alerts, CCTV integration, and dashboards can be added in the next phase.")

tab_image, tab_about = st.tabs(["Detection", "About"])

with tab_image:
    col_input, col_output = st.columns(2, gap="large")

    with col_input:
        st.subheader("Input")
        source = st.radio(
            "Input source",
            ["Sample image", "Upload image", "Camera"],
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
        elif source == "Upload image":
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
        else:
            camera = st.camera_input("Capture from camera")
            if camera:
                input_image = decode_image(camera.getvalue())
                source_label = "Camera capture"
                st.image(
                    cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB),
                    caption="Camera capture",
                    use_container_width=True,
                )

    with col_output:
        st.subheader("Result")

        if input_image is None:
            st.info("Select a sample, upload an image, or take a photo to run detection.")
        else:
            with st.spinner("Analyzing frame..."):
                result = detect(input_image, get_model(), conf=confidence)

            render_metrics(result)
            st.image(result.annotated_rgb, caption="Annotated result", use_container_width=True)

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

with tab_about:
    st.subheader("About this demo")
    st.markdown(
        """
        This module is part of a **Safety & Security Analytics** platform. It uses a
        YOLO11 model to identify smoking activity in images and camera captures.

        **Current capabilities**
        - Real-time image analysis
        - Bounding boxes with confidence scores
        - Downloadable evidence image for client review

        **Planned next phase**
        - Email / SMS / WhatsApp alerts
        - Automatic incident snapshots
        - CCTV stream integration
        - Admin dashboard and audit logs
        """
    )
