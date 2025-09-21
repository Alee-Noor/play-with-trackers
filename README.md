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


## Customization
Modify the parameters at the bottom of the script to customize the behavior:
- VIDEO_SOURCE = "v_test5.mp4"        # 0 for webcam, or path to video file
- MOT_METHOD = 'bytetrack'            # 'bytetrack' or 'deepsort'
- SOT_TRACKER = 'csrt'                # 'csrt', 'kcf', or 'mosse'
- YOLO_MODEL = 'yolov8n.pt'           # YOLO model file
- TARGET_CLASSES = [2,3,4,5,6,7]      # COCO dataset class IDs to track
- CONFIDENCE_THRESHOLD = 0.45         # Detection confidence threshold
- SAVE_OUTPUT = True                  # Enable/disable video recording
- OUTPUT_PATH = "tracked_output.mp4"  # Output video path


## 🎮 Controls
- Click on object → Switch to Single Object Tracking mode
- ESC key → Return to MOT mode (from SOT)
- S key → Save current frame as an image
- Q key → Quit


## ⚙️ Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/motion-tracking-app.git
   cd motion-tracking-app

2. Create a virtual environment (recommended):
   
   - python -m venv venv
   # Linux/Mac:
   - source venv/bin/activate
   # Windows  
   - venv\Scripts\activate

3.Install dependencies:

  - pip install -r requirements.txt

4. Just Run the main file code: main32.py



