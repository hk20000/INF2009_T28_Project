# Classroom Assistant - Real-Time Engagement & Speech Transcription

##  Project Overview
This project is designed as a real-time classroom assistant to help teachers monitor student engagement during lessons. By leveraging speech-to-text transcription, facial recognition, and an interactive dashboard, the system provides real-time insights into classroom dynamics. 

---

##  Features
-  **Face Recognition**: Detects and identifies students.
-  **Expression Detection**: Recognizes engagement levels (e.g., engaged, distracted, talking).
-  **Speech-to-Text**: Transcribes lectures in real-time.
-  **Dashboard**: Displays engagement trends and insights.
---

## Items needed:
1. Raspberry Pi 4 (or later) with Raspberry Pi OS (64-bit recommended).

2. Microphone (USB or 3.5mm jack) for Speech Recognition.

3. Webcam (e.g., Logitech C270) for Face Recognition.

4. Stable internet connection for downloading models and packages.

##  Installation & Setup (Raspberry Pi)

### 📌 Prerequisites
Ensure your Raspberry Pi has **Python 3** and **pip** installed:
```bash
sudo apt update && sudo apt upgrade -y
sudo apt install python3 python3-pip -y
```

### 🎙 Setting Up Speech-to-Text Transcription
1️⃣ Install dependencies:
```bash
pip3 install vosk pyaudio numpy
```

2️⃣ Download a lightweight Vosk speech model:
```bash
mkdir -p ~/vosk_models && cd ~/vosk_models
wget https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip
unzip vosk-model-small-en-us-0.15.zip
mv vosk-model-small-en-us-0.15 vosk-model
```

3️⃣ Update system libraries:
```bash
sudo apt update
sudo apt install -y portaudio19-dev python3-pyaudio
pip3 install pyaudio
```

---

### 🎭 Setting Up Face Recognition & Dashboard
1️⃣ Update Raspberry Pi:
```bash
sudo apt update && sudo apt full-upgrade -y
```

2️⃣ Install necessary Python packages:
```bash
pip install opencv-python imutils face-recognition streamlit sqlite3 pandas plotly openai requests httpx
```

3️⃣ Install CMake (required for face recognition):
```bash
sudo apt install cmake -y
```

---

## 🔧 Running the System
###  Run Speech-to-Text Transcription
```bash
python3 speech_to_sub.py
```

###  Run Real-Time Face Recognition
```bash
python3 face.py
```

###  Run Engagement Dashboard
```bash
streamlit run dashboard.py
```


## 💡 Methodology & Design Choices
### 📷 Face Recognition & Engagement Detection
- **Library Used:** `face_recognition`
- **Why?** Lightweight & accurate for real-time processing on Raspberry Pi.
- **Expression Analysis:** Uses `mediapipe` for facial landmark detection.

### 🗣 Speech-to-Text
- **Library Used:** `Vosk`
- **Why?** Offline speech recognition for efficiency & privacy.
- **Model Used:** `vosk-model-small-en-us-0.15` (lightweight & optimized for Raspberry Pi).

### 📊 Engagement Dashboard
- **Built With:** `Streamlit` for real-time visualization.
