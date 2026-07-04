import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import joblib
import mediapipe as mp
import time

st.title("Sign Language Recognition")
st.write("Live Sign Language Word Prediction")

model = joblib.load("model/sign_language_landmark_model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

word = ""
last_letter = ""
last_time = time.time()

def video_frame_callback(frame):
    global word, last_letter, last_time

    img = frame.to_ndarray(format="bgr24")
    img = cv2.flip(img, 1)

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    current_letter = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []

            for point in hand_landmarks.landmark:
                landmarks.append(point.x)
                landmarks.append(point.y)

            if len(landmarks) == 42:
                current_letter = str(model.predict([landmarks])[0])

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            now = time.time()
            if current_letter and current_letter != last_letter and now - last_time > 1.5:
                word += current_letter
                last_letter = current_letter
                last_time = now

    cv2.putText(img, "Letter: " + current_letter, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.putText(img, "Word: " + word, (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3)

    return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="sign-language-live-word",
    video_frame_callback=video_frame_callback
)