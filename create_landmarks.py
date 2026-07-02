import os
import cv2
import pickle
import mediapipe as mp

DATA_DIR = "dataset"

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

data = []
labels = []

for folder in os.listdir(DATA_DIR):
    folder_path = os.path.join(DATA_DIR, folder)

    if not os.path.isdir(folder_path):
        continue

    label = folder.split("-")[0]

    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)

        if img is None:
            continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand = result.multi_hand_landmarks[0]
            landmarks = []

            for point in hand.landmark:
                landmarks.append(point.x)
                landmarks.append(point.y)

            data.append(landmarks)
            labels.append(label)

with open("model/data.pickle", "wb") as f:
    pickle.dump({"data": data, "labels": labels}, f)

print("Landmark data created successfully")
print("Total samples:", len(data))