import streamlit as st
import numpy as np
import cv2
from pathlib import Path
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

try:
    net, clf = load_models()
except Exception as e:
    st.error(f"Model load error: {e}")
    st.stop()

# ---------- Inference ----------
def detect_and_annotate(bgr_img: np.ndarray, conf_thresh: float = 0.5):
    h, w = bgr_img.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(bgr_img, (300, 300)), 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    dets = net.forward()

    count = 0
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
        color = (0, 200, 0) if label == "Mask" else (0, 0, 200)

        cv2.rectangle(bgr_img, (x1, y1), (x2, y2), color, 2)
        cv2.putText(bgr_img, f"{label}: {score:.2f}", (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        count += 1

    return bgr_img, count

# ---------- UI ----------
st.sidebar.header("Settings")
conf_thresh = st.sidebar.slider("Face detection confidence", 0.1, 0.9, 0.5, 0.05)
mode = st.radio("Choose input", ["Upload Image", "Webcam (beta)"])

if mode == "Upload Image":
    up = st.file_uploader("Upload a JPG/PNG", type=["jpg", "jpeg", "png"])
    if up:
        data = np.frombuffer(up.read(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read image.")
        else:
            out, n = detect_and_annotate(bgr.copy(), conf_thresh)
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            st.caption(f"Faces processed: {n}")

else:
    st.info("Click below and allow camera access in your browser.")
    cam = st.camera_input("Live camera")
    if cam is not None:
        data = np.frombuffer(cam.getvalue(), np.uint8)
        bgr = cv2.imdecode(data, cv2.IMREAD_COLOR)
        if bgr is None:
            st.error("Could not read camera frame.")
        else:
            out, n = detect_and_annotate(bgr.copy(), conf_thresh)
            st.image(cv2.cvtColor(out, cv2.COLOR_BGR2RGB), channels="RGB", use_column_width=True)
            st.caption(f"Faces processed: {n}")
