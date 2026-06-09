# 🔊 Gesture Volume Controller Using MediaPipe

Control your system volume using hand gestures and a webcam.

This project uses MediaPipe Hand Tracking to detect the distance between the thumb and index finger. Based on the distance, the application automatically increases or decreases the system volume using PyAutoGUI.

## 🚀 Features

* Real-time hand tracking
* Gesture-based volume control
* Volume Up gesture
* Volume Down gesture
* Visual feedback with landmarks
* Lightweight and easy to run

## 🛠️ Technologies Used

* Python
* OpenCV
* MediaPipe
* PyAutoGUI

## 📂 Project Structure

```text
Day04-GestureVolumeController/
│
├── main.py
├── requirements.txt
├── README.md
└── screenshots/
```

## 📦 Installation

Clone the repository:

```bash
git clone <repository-url>
cd Day04-GestureVolumeController
```

Install dependencies:

```bash
pip install mediapipe opencv-python pyautogui
```

## ▶️ Run the Project

```bash
python main.py
```

## ✋ Hand Landmarks Used

| Landmark | Description      |
| -------- | ---------------- |
| 4        | Thumb Tip        |
| 8        | Index Finger Tip |

## 🎯 Gestures

### Volume Up

Move the thumb and index finger far apart.

```text
Thumb  ------------------  Index
```

### Volume Down

Bring the thumb and index finger close together.

```text
Thumb -- Index
```

## ⚙️ How It Works

1. Capture webcam frames using OpenCV.
2. Detect hand landmarks using MediaPipe.
3. Calculate the distance between:

   * Thumb Tip (Landmark 4)
   * Index Finger Tip (Landmark 8)
4. Compare the distance with predefined thresholds.
5. Trigger system volume keys using PyAutoGUI.

## 📚 Concepts Learned

* MediaPipe Hand Tracking
* Landmark Detection
* Gesture Recognition
* Coordinate Systems
* Distance Calculation
* Human Computer Interaction (HCI)

## 🔮 Future Improvements

* Mute / Unmute Gesture
* Brightness Control Mode
* Two-Hand Gesture Support
* Gesture Sensitivity Settings
* GUI Dashboard
* Smooth Volume Transitions

## 📸 Demo

Add screenshots or GIFs here.

```text
screenshots/demo.gif
```

## 🤝 Contributing

Feel free to fork the repository and improve the project.

## ⭐ Support

If you found this project useful, consider giving it a star.

## #30DaysOfMediaPipe

Day 4 of the 30 Days of MediaPipe Hand Projects challenge.
