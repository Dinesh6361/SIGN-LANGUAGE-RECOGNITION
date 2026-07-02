import cv2
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("model/sign_language_model.h5")

with open("model/labels.txt", "r") as f:
    labels = [line.strip() for line in f.readlines()]

IMG_SIZE = 64

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.rectangle(frame, (100, 100), (400, 400), (0, 255, 0), 2)

    roi = frame[100:400, 100:400]
    img = cv2.resize(roi, (IMG_SIZE, IMG_SIZE))
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    index = np.argmax(prediction)
    sign = labels[index]

    cv2.putText(frame, f"Sign: {sign}", (100, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Sign Language Recognition", frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()