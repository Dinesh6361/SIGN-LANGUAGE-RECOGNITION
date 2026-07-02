import cv2
import time
import joblib
import mediapipe as mp

model = joblib.load("model/sign_language_landmark_model.pkl")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)

word = ""
last_letter = ""
last_time = time.time()

while True:
    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    result = hands.process(rgb)

    current_letter = ""

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            landmarks = []

            for point in hand_landmarks.landmark:
                landmarks.append(point.x)
                landmarks.append(point.y)

            prediction = model.predict([landmarks])[0]
            current_letter = prediction

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            current_time = time.time()

            if current_letter != last_letter and current_time - last_time > 1.5:
                word += current_letter
                last_letter = current_letter
                last_time = current_time

    cv2.putText(frame, "Letter: " + current_letter, (30, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, "Word: " + word, (30, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    cv2.putText(frame, "Press SPACE=space | B=backspace | C=clear | Q=quit",
                (30, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    cv2.imshow("Sign Language Word Prediction", frame)

    key = cv2.waitKey(1)

    if key == ord(" "):
        word += " "

    elif key == ord("b"):
        word = word[:-1]

    elif key == ord("c"):
        word = ""

    elif key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()