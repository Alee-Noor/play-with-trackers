# play-with-trackers
A simple Single and Multi Object tracker system for drone espacially for Vehicles

# Simplified Motion Tracking App

This is a Python-based **object tracking application** that uses **YOLOv8** for object detection and supports two multi-object tracking (MOT) methods:
- **ByteTrack** (built into Ultralytics YOLO)
- **DeepSORT** (via [deep-sort-realtime](https://github.com/levan92/deep-sort-realtime))

Additionally, you can switch to **Single Object Tracking (SOT)** interactively with OpenCV trackers (**CSRT**, **KCF**, or **MOSSE**).  
The app also supports **video recording** with overlays and a **transparent info panel**.

---

## 🚀 Features
- Real-time **Multi-Object Tracking** (MOT) using YOLOv8 + ByteTrack or DeepSORT
- Interactive **Single Object Tracking (SOT)** (click on object to follow it)
- **Video recording** to MP4 with bounding boxes and overlays
- Transparent **info panel** with FPS, frame count, and status
- Save frames manually with the `S` key
- Works with **webcam** or **video files**

---

## 📦 Requirements
- Python **3.8+**  
- OpenCV (with `contrib` for SOT trackers)  
- Ultralytics YOLOv8  
- DeepSORT realtime tracker  
- NumPy  

All dependencies are listed in `requirements.txt`.

---

## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/motion-tracking-app.git
   cd motion-tracking-app

