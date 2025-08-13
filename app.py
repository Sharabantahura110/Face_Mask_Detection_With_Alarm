import streamlit as st
import numpy as np
import cv2
from pathlib import Path
from base64 import b64encode

from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

st.set_page_config(page_title="Face Mask Detector", page_icon="😷", layout="centered")
st.title("😷 Face Mask Detection")

# ---------- Load models once (cached) ----------
@st.cache_resource
def load_models():
    face_proto = Path("face_detector") / "deploy.prototxt"
    face_model = Path("face_detector") / "res10_300x300_ssd_iter_140000.caffemodel"
    if not face_proto.exists() or not face_model.exists():
        raise FileNotFoundError(
            "Face detector files not found. Make sure 'face_detector/deploy.prototxt' "
            "and 'face_detector/res10_300x300_ssd_iter_140000.caffemodel' exist in the repo."
        )
    net = cv2.dnn.readNetFromCaffe(str(face_proto), str(face_model))

    mask_model_path = Path("mask_detector.model")
    if not mask_model_path.exists():
        raise FileNotFoundError(
            "Mask model not found. Place 'mask_detector.model' at the repo root."
        )
    clf = load_model(str(mask_model_path))
    return net, clf

# Load optional alarm sound (once)
@st.cache_resource
def load_alarm():
    p = Path("alert.mp3")
    if not p.exists():
        return None, None
    data = p.read_bytes()
    return data, b64encode(data).decode()

try:
    net, clf = load_models()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

alarm_bytes, alarm_b64 = load_alarm()

# ---------- Inference ----------
def detect_and_annotate(bgr_img: np.ndarray, conf_thresh: float = 0.5):
    """Detect faces and classify mask vs no-mask; draw boxes; return flags for alarm."""
    h, w = bgr_img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets = net.forward()

    count = 0
    has_mask = False
    has_no_mask = False

    for i in range(dets.shape[2]):
        confidence = dets[0, 0, i, 2]
        if confidence < conf_thresh:
            continue

        box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
        (x1, y1, x2, y2) = box.astype("int")
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w - 1, x2), min(h - 1, y2)

        face = bgr_img[y1:y2, x1:x2]
        if face.size == 0:
            continue

        face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224))
        arr = img_to_array(face_rgb)
        arr = preprocess_input(arr)
        arr = np.expand_dims(arr, axis=0)

        (mask, no_mask) = clf.predict(arr, verbose=0)[0]
        label = "Mask" if mask > no_mask else "No Mask"
        score = float(max(mask, no_mask))

        if label == "Mask":
            has_mask = True
            color = (0, 200, 0)
        else:
            has_no_mask = True
            color = (0, 0, 200)

        cv2.rectangle(bgr_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr_img, f"{label}: {score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        count += 1

    return bgr_img, count, has_mask, has_no_mask

# ---------- UI ----------
st.sidebar.header("Settings")
conf_thresh = st.sidebar.slider("Face detection confidence", 0.1, 0.9, 0.5, 0.05)

# Alarm controls
alarm_enabled = st.sidebar.toggle("Enable alarm sound", value=True)
alarm_trigger = st.sidebar.selectbox("Alarm when …", ["No Mask", "Mask"])
if alarm_enabled and alarm_bytes is None:
    st.sidebar.warning("alert.mp3 not found at repo root — sound will be disabled.")

mode = st.radio("Choose input", ["Upload Image", "Webcam (beta)"])

def maybe_play_alarm(has_mask: bool, has_no_mask: bool):
    """Play alarm if condition met; show visual notice. Browser may block auto-play."""
    if not alarm_enabled or alarm_bytes is None:
        return
    should_alarm = (alarm_trigger == "No Mask" and has_no_mask) or (alarm_trigger == "Mask" and has_mask)
    if should_alarm:
        st.warning("🚨 Alarm triggered")
        # Guaranteed playable after interaction:
        st.audio(alarm_bytes, format="audio/mp3")
        # Best-effort autoplay (may be blocked until user interacts):
        st.markdown(
            f"""
            <audio autoplay>
              <source src="data:audio/mp3;base64,{alarm_b64}" type="audio/mpeg">
            </audio>
            """,
            unsafe_allow_html=True
        )

if mode == "Upload Image":
    up = st.file_uploader("Upload a JPG/PNG", type=["jpg", "jpeg", "png"])
    if up:
        data = np.frombuffer(up.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read image.")
        else:
            out, n, has_mask, has_no_mask = detect_and_annotate(bgr.copy(), conf_thresh)
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            st.caption(f"Faces processed: {n}")
            maybe_play_alarm(has_mask, has_no_mask)

else:
    st.info("Click below and allow camera access in your browser.")
    cam = st.camera_input("Live camera")
    if cam is not None:
        data = np.frombuffer(cam.getvalue(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read camera frame.")
        else:
            out, n, has_mask, has_no_mask = detect_and_annotate(bgr.copy(), conf_thresh)
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            st.caption(f"Faces processed: {n}")
            maybe_play_alarm(has_mask, has_no_mask)
