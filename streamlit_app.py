from datetime import datetime
from io import BytesIO
from pathlib import Path

import av
import cv2
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoTransformerBase, webrtc_streamer

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


def render_live_metrics(transformer: "SmokingDetector"):
    c1, c2, c3 = st.columns(3)
    status = "ALERT" if transformer.last_count else "CLEAR"
    c1.metric("Live status", status)
    c2.metric("Detections", transformer.last_count)
    c3.metric("Inference", f"{transformer.last_inference_ms:.0f} ms")

    if transformer.last_count:
        st.error(f"Smoking activity detected — {transformer.last_count} instance(s)")
    else:
        st.success("Monitoring… no smoking detected in current frame")


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


def render_live_detection_table(transformer: "SmokingDetector"):
    if not transformer.last_detections:
        st.info("No smoking activity detected in the current live frame.")
        return

    rows = [
        {
            "#": index,
            "Label": item.label,
            "Confidence": f"{item.confidence * 100:.1f}%",
            "Box (x1, y1, x2, y2)": f"({item.x1}, {item.y1}, {item.x2}, {item.y2})",
        }
        for index, item in enumerate(transformer.last_detections, start=1)
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


class SmokingDetector(VideoTransformerBase):
    def __init__(self, conf: float = 0.25):
        self.conf = conf
        self.model = get_model()
        self.frame_index = 0
        self.detect_every = 2
        self.last_annotated = None
        self.last_count = 0
        self.last_inference_ms = 0.0
        self.last_detections = []

    def transform(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="bgr24")
        self.frame_index += 1

        if self.frame_index % self.detect_every == 0:
            result = detect(image, self.model, conf=self.conf)
            self.last_annotated = result.annotated_bgr
            self.last_count = result.count
            self.last_inference_ms = result.inference_ms
            self.last_detections = result.detections

        if self.last_annotated is not None:
            return av.VideoFrame.from_ndarray(self.last_annotated, format="bgr24")
        return frame


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
        "1. Open **Live CCTV** tab\n"
        "2. Click **Start** on the camera\n"
        "3. Allow browser camera access\n"
        "4. Or use **Image Analysis** for upload/samples"
    )
    st.divider()
    st.caption("Proof-of-concept demo. Alerts, CCTV integration, and dashboards can be added in the next phase.")

tab_live, tab_image, tab_about = st.tabs(["Live CCTV", "Image Analysis", "About"])

with tab_live:
    st.subheader("Live monitoring")
    st.caption("Works like a CCTV feed — detection runs continuously while the camera is active.")

    def create_detector():
        return SmokingDetector(conf=confidence)

    webrtc_ctx = webrtc_streamer(
        key="live-cctv",
        video_transformer_factory=create_detector,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

    st.info("Click **Start** above, allow camera access, and monitoring begins automatically.")

    if webrtc_ctx.state.playing and webrtc_ctx.video_transformer:
        transformer = webrtc_ctx.video_transformer
        render_live_metrics(transformer)
        st.markdown("#### Live detection details")
        render_live_detection_table(transformer)
    elif webrtc_ctx.state.playing:
        st.warning("Camera started. Initializing detector…")
    else:
        st.info("Camera is off. Press **Start** to begin live monitoring.")

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
        YOLO11 model to identify smoking activity in live camera feeds and images.

        **Current capabilities**
        - Live CCTV-style camera monitoring
        - Image upload and sample testing
        - Bounding boxes with confidence scores
        - Downloadable evidence image for client review

        **Planned next phase**
        - Email / SMS / WhatsApp alerts
        - Automatic incident snapshots
        - RTSP CCTV stream integration
        - Admin dashboard and audit logs
        """
    )
