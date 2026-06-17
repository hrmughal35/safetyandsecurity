from pathlib import Path

import cv2
import numpy as np
import streamlit as st
from ultralytics import YOLO

MODEL_PATH = Path(__file__).resolve().parent / "best.onnx"


@st.cache_resource
def load_model():
    return YOLO(str(MODEL_PATH))


def run_detection(image_bgr: np.ndarray):
    model = load_model()
    results = model(image_bgr)
    result = results[0]
    annotated = result.plot()
    count = len(result.boxes) if result.boxes is not None else 0
    return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), count


def decode_image(file_bytes: bytes) -> np.ndarray:
    nparr = np.frombuffer(file_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("Could not read image.")
    return image


st.set_page_config(page_title="Smoking Detection", page_icon="🚭", layout="wide")

st.title("Smoking Detection Demo")
st.caption("YOLO11-based smoking activity detection for safety & security analytics.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Input")
    source = st.radio("Choose input", ["Upload image", "Camera"], horizontal=True)

    input_image = None
    if source == "Upload image":
        uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png", "webp"])
        if uploaded:
            input_image = decode_image(uploaded.read())
            st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), caption="Uploaded image", use_container_width=True)
    else:
        camera = st.camera_input("Take a photo")
        if camera:
            input_image = decode_image(camera.getvalue())
            st.image(cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB), caption="Camera capture", use_container_width=True)

with col2:
    st.subheader("Result")
    if input_image is not None:
        with st.spinner("Running detection..."):
            annotated_rgb, count = run_detection(input_image)

        st.image(annotated_rgb, caption="Detection result", use_container_width=True)

        if count > 0:
            st.error(f"Smoking detected: {count} instance(s)")
        else:
            st.success("No smoking detected")
    else:
        st.info("Upload an image or take a photo to run detection.")

st.divider()
st.markdown(
    "**Demo notes:** This is a proof-of-concept. "
    "Results depend on image quality, angle, and lighting."
)
