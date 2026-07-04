import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import joblib
import numpy as np
import os

st.title("Sign Language Recognition")
st.write("Webcam Live Demo")

MODEL_PATH = "model/sign_language_landmark_model.pkl"
LABELS_PATH = "model/labels.txt"

@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = joblib.load(MODEL_PATH)

    with open(LABELS_PATH, "r") as f:
        labels = [line.strip() for line in f.readlines()]

    return model, labels

model, labels = load_model()

st.success("Model loaded successfully!")

def video_frame_callback(frame):
    img = frame.to_ndarray(format="bgr24")

    cv2.putText(img, "Camera Working", (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="sign-language-camera",
    video_frame_callback=video_frame_callback
)