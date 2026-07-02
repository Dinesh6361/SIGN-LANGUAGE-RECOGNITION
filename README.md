# 🤟 Real-Time Sign Language Recognition using Python

A real-time Sign Language Recognition system that uses **Python**, **OpenCV**, **MediaPipe**, and **Machine Learning** to recognize hand gestures from a webcam, convert them into text, form words/sentences, and provide speech output.

---

## 📌 Features

- 🎥 Real-time webcam detection
- ✋ Hand landmark detection using MediaPipe
- 🔤 Alphabet (A-Z) recognition
- 📝 Automatic word formation
- 💬 Sentence formation
- 🔊 Text-to-Speech conversion
- 📈 High accuracy gesture recognition
- ⚡ Fast real-time prediction
- 🖥️ Easy-to-use interface

---

## 🛠️ Technologies Used

- Python 3.10
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- TensorFlow (for deep learning models)
- Joblib
- Pyttsx3

---

## 📂 Project Structure

```
Sign-Language-Recognition/
│
├── dataset/
├── dynamic_dataset/
├── model/
├── dynamic_model/
│
├── create_landmarks.py
├── train_landmark_model.py
├── predict_words.py
├── collect_dynamic_data.py
├── train_dynamic_model.py
├── predict_dynamic_words.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## 🚀 Installation

### Clone Repository

```bash
git clone https://github.com/Dinesh6361/SIGN-LANGUAGE-RECOGNITION.git
```

```bash
cd SIGN-LANGUAGE-RECOGNITION
```

### Create Virtual Environment

```bash
python -m venv venv
```

### Activate Environment

Windows

```bash
venv\Scripts\activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Run the Project

### Generate Landmark Dataset

```bash
python create_landmarks.py
```

### Train the Model

```bash
python train_landmark_model.py
```

### Start Real-Time Prediction

```bash
python predict_words.py
```

---

## 📷 Workflow

```
Webcam
      │
      ▼
MediaPipe Hand Detection
      │
      ▼
Hand Landmarks
      │
      ▼
Machine Learning Model
      │
      ▼
Letter Prediction
      │
      ▼
Word Formation
      │
      ▼
Sentence Formation
      │
      ▼
Text-to-Speech
```

---

## 🎯 Future Enhancements

- Dynamic gesture recognition
- Indian Sign Language (ISL) support
- Complete sentence recognition
- Translation to multiple languages
- Web application
- Mobile application
- Cloud deployment

---

## 📊 Applications

- Communication assistance for deaf and mute individuals
- Educational institutions
- Healthcare
- Public service centers
- Smart accessibility systems

---

## 👨‍💻 Author

**Dinesh S**

📧 Email: dineshs200315@gmail.com

🔗 GitHub: https://github.com/Dinesh6361

---

## ⭐ If you found this project useful

Please consider giving this repository a **Star ⭐** on GitHub.
