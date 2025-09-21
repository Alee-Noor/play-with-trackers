"""
Simplified Motion Tracking App - Works with ByteTrack or DeepSORT
Dependencies: opencv-python, ultralytics, numpy, deep-sort-realtime
Added features: Video recording and transparent info background
"""

import cv2
import numpy as np
from ultralytics import YOLO
import time
import os

# Import DeepSORT (installed via pip install deep-sort-realtime)
from deep_sort_realtime.deepsort_tracker import DeepSort


class SimpleMotionTracker:
    def __init__(self,
                 video_source=0,
                 yolo_model='yolov8n.pt',
                 mot_method='bytetrack',  # 'bytetrack' or 'deepsort'
                 sot_tracker='kcf',  # 'csrt', 'kcf', or 'mosse'
                 target_classes=None,
                 confidence_threshold=0.5,
                 save_output=True,  # New parameter for video recording
                 output_path='output_tracking.mp4'):  # New parameter for output path
        """
        Initialize the motion tracker with configuration parameters
        
        Args:
            video_source: Input video source (0 for webcam, or path to video file)
            yolo_model: Path to YOLO model weights file
            mot_method: Multi-object tracking method ('bytetrack' or 'deepsort')
            sot_tracker: Single object tracking algorithm ('csrt', 'kcf', or 'mosse')
            target_classes: List of class IDs to track (None for all classes)
            confidence_threshold: Minimum confidence for detection
            save_output: Whether to save output video
            output_path: Path for output video file
        """
        # Configuration
        self.video_source = video_source
        self.mot_method = mot_method.lower()
        self.sot_tracker_type = sot_tracker.lower()
        self.target_classes = target_classes
        self.confidence_threshold = confidence_threshold
        self.save_output = save_output
        self.output_path = output_path

        # Initialize YOLO object detection model
        print("Loading YOLO model...")
        self.yolo = YOLO(yolo_model)

        # Initialize Multi-Object Tracking (MOT) tracker
        if self.mot_method == "deepsort":
            # Use DeepSORT tracker
            self.mot_tracker = DeepSort(max_age=30, n_init=3)
        else:
            # Use ByteTrack via ultralytics (handled in detect_objects method)
            self.mot_tracker = None

        # Single Object Tracking (SOT) variables
        self.sot_mode = False  # Flag to indicate if in single object tracking mode
        self.sot_tracker = None  # SOT tracker instance
        self.sot_bbox = None  # Bounding box of the tracked object
        self.sot_id = None  # ID of the tracked object

        # Mouse callback variables for interactive object selection
        self.mouse_x = 0
        self.mouse_y = 0
        self.mouse_clicked = False

        # Visualization settings
        self.frame_count = 0  # Counter for processed frames
        np.random.seed(42)  # Set seed for consistent color generation
        self.colors = np.random.randint(0, 255, size=(100, 3), dtype=np.uint8)  # Colors for different tracks

        # FPS calculation variables
        self.fps_start_time = time.time()
        self.fps_frame_count = 0
        self.fps = 0

        # Video recording variables
        self.video_writer = None
        self.recording = False

    def _init_video_writer(self, frame_width, frame_height, fps=30.0):
        """
        Initialize video writer for recording output
        
        Args:
            frame_width: Width of the output video frames
            frame_height: Height of the output video frames
            fps: Frames per second for the output video
        """
        if self.save_output:
            # Create output directory if it doesn't exist
            os.makedirs(os.path.dirname(self.output_path) if os.path.dirname(self.output_path) else '.', exist_ok=True)
            
            # Define codec and create VideoWriter
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # MP4 video codec
            self.video_writer = cv2.VideoWriter(
                self.output_path, 
                fourcc, 
                fps, 
                (frame_width, frame_height)
            )
            # Check if video writer initialized successfully
            if self.video_writer.isOpened():
                self.recording = True
                print(f"Recording started: {self.output_path}")
            else:
                print(f"Failed to initialize video writer for {self.output_path}")

    def _init_sot_tracker(self, frame, bbox):
        """
        Initialize Single Object Tracker with the selected algorithm
        
        Args:
            frame: Initial frame to initialize the tracker
            bbox: Bounding box of the object to track [x1, y1, x2, y2]
            
        Returns:
            Initialized tracker object
        """
        tracker = None

        try:
            # Initialize the selected tracker type
            if self.sot_tracker_type == 'csrt':
                # CSRT tracker (high accuracy but slower)
                if hasattr(cv2, 'TrackerCSRT_create'):
                    tracker = cv2.TrackerCSRT_create()
                else:
                    tracker = cv2.legacy.TrackerCSRT_create()
            elif self.sot_tracker_type == 'kcf':
                # KCF tracker (good balance of speed and accuracy)
                if hasattr(cv2, 'TrackerKCF_create'):
                    tracker = cv2.TrackerKCF_create()
                else:
                    tracker = cv2.legacy.TrackerKCF_create()
            elif self.sot_tracker_type == 'mosse':
                # MOSSE tracker (very fast but less accurate)
                if hasattr(cv2, 'TrackerMOSSE_create'):
                    tracker = cv2.TrackerMOSSE_create()
                else:
                    tracker = cv2.legacy.TrackerMOSSE_create()
        except AttributeError:
            # Fallback if selected tracker is not available
            print(f"Tracker {self.sot_tracker_type} not available in your OpenCV version")
            if hasattr(cv2, 'TrackerKCF_create'):
                tracker = cv2.TrackerKCF_create()
            else:
                tracker = cv2.legacy.TrackerKCF_create()

        # Check if tracker was initialized successfully
        if tracker is None:
            raise ValueError(f"Could not initialize tracker: {self.sot_tracker_type}")

        # Convert bounding box from [x1, y1, x2, y2] to [x, y, width, height] format
        x, y, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]-bbox[0]), int(bbox[3]-bbox[1])
        # Initialize tracker with the object bounding box
        tracker.init(frame, (x, y, w, h))
        return tracker

    def mouse_callback(self, event, x, y, flags, param):
        """
        Mouse callback function for interactive object selection
        
        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Additional flags
            param: Additional parameters
        """
        self.mouse_x = x
        self.mouse_y = y
        # Set flag when left mouse button is clicked
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_clicked = True

    def check_click_on_bbox(self, bboxes):
        """
        Check if a mouse click occurred on any bounding box
        
        Args:
            bboxes: List of bounding boxes to check
            
        Returns:
            Index of the clicked bounding box or None if no box was clicked
        """
        if not self.mouse_clicked:
            return None
        # Check each bounding box for intersection with mouse click
        for i, bbox in enumerate(bboxes):
            x1, y1, x2, y2 = bbox[:4]
            if x1 <= self.mouse_x <= x2 and y1 <= self.mouse_y <= y2:
                self.mouse_clicked = False
                return i
        self.mouse_clicked = False
        return None

    def detect_objects(self, frame):
        """
        Detect and track objects in the frame using the selected method
        
        Args:
            frame: Input frame to process
            
        Returns:
            List of detected and tracked objects with their properties
        """
        detections = []

        if self.mot_method == "bytetrack":
            # Use YOLO's built-in ByteTrack for detection and tracking
            results = self.yolo.track(frame, conf=self.confidence_threshold,
                                      persist=True, verbose=False)
            # Process detection results
            for r in results:
                if hasattr(r, "boxes") and r.boxes is not None:
                    for box in r.boxes:
                        # Extract bounding box coordinates
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])  # Confidence score
                        cls = int(box.cls[0])      # Class ID
                        track_id = int(box.id[0]) if box.id is not None else -1  # Track ID
                        # Add to detections if class is in target classes (or all if None)
                        if self.target_classes is None or cls in self.target_classes:
                            detections.append({
                                "id": track_id,
                                "bbox": [x1, y1, x2, y2],
                                "conf": conf,
                                "cls": cls
                            })

        elif self.mot_method == "deepsort":
            # Run YOLO detection first, then track with DeepSORT
            results = self.yolo(frame, conf=self.confidence_threshold, verbose=False)
            bboxes = []
            # Extract detections from YOLO results
            for r in results:
                if r.boxes is not None:
                    for box in r.boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0])
                        cls = int(box.cls[0])
                        # Convert to [x, y, width, height] format for DeepSORT
                        if self.target_classes is None or cls in self.target_classes:
                            bboxes.append(([x1, y1, x2-x1, y2-y1], conf, cls))

            # Update DeepSORT tracker with detections
            tracks = self.mot_tracker.update_tracks(bboxes, frame=frame)
            # Process tracking results
            for t in tracks:
                if not t.is_confirmed():
                    continue  # Skip unconfirmed tracks
                ltrb = t.to_ltrb()  # Convert to [left, top, right, bottom] format
                detections.append({
                    "id": t.track_id,
                    "bbox": ltrb,
                    "conf": 1.0,  # DeepSORT doesn't provide confidence, set to 1.0
                    "cls": None   # DeepSORT doesn't provide class information
                })

        return detections

    def update_fps(self):
        """Calculate and update frames per second (FPS)"""
        self.fps_frame_count += 1
        # Update FPS every 30 frames
        if self.fps_frame_count >= 30:
            end_time = time.time()
            self.fps = self.fps_frame_count / (end_time - self.fps_start_time)
            self.fps_start_time = end_time
            self.fps_frame_count = 0

    def draw_tracks(self, frame, tracks):
        """
        Draw bounding boxes and labels for tracked objects
        
        Args:
            frame: Frame to draw on
            tracks: List of tracked objects
            
        Returns:
            Frame with drawn tracks
        """
        for track in tracks:
            track_id = track['id']
            bbox = track['bbox']
            conf = track.get('conf', 0)
            # Get a consistent color for this track ID
            color = self.colors[int(track_id) % 100].tolist()
            # Convert bounding box coordinates to integers
            x1, y1, x2, y2 = [int(x) for x in bbox]
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            # Create label with ID and confidence
            label = f"ID:{track_id} ({conf:.2f})"
            # Draw label above bounding box
            cv2.putText(frame, label, (x1, max(y1-10, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        return frame

    def draw_sot_bbox(self, frame, bbox):
        """
        Draw bounding box for single object tracking
        
        Args:
            frame: Frame to draw on
            bbox: Bounding box coordinates
            
        Returns:
            Frame with drawn bounding box
        """
        if bbox is not None:
            x1, y1, x2, y2 = [int(x) for x in bbox]
            # Draw green bounding box for single object tracking
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            # Draw label with tracker type and object ID
            cv2.putText(frame, f"SOT: {self.sot_tracker_type.upper()} ID:{self.sot_id}",
                        (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return frame

    def draw_info(self, frame):
        """
        Draw information overlay with transparent background
        
        Args:
            frame: Frame to draw on
            
        Returns:
            Frame with information overlay
        """
        info_text = []
        # Add mode information
        if self.sot_mode:
            info_text.append(f"Mode: Single Object Tracking ({self.sot_tracker_type.upper()})")
            info_text.append("Press 'ESC' to return to MOT")
        else:
            info_text.append(f"Mode: Multi-Object Tracking ({self.mot_method.upper()})")
            info_text.append("Click on any object to track it")

        # Add performance and status information
        info_text.append(f"FPS: {self.fps:.1f}")
        info_text.append(f"Frame: {self.frame_count}")
        
        # Add recording status
        if self.recording:
            info_text.append(f"Recording: {self.output_path}")
        
        # Add control instructions
        info_text.append("Press 'Q' to quit | 'S' to save frame")

        # Calculate background rectangle dimensions
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        margin = 8
        line_height = 20
        
        # Get max text width
        max_width = 0
        for text in info_text:
            (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
            max_width = max(max_width, text_width)
        
        # Calculate background rectangle size
        bg_width = max_width + 2 * margin
        bg_height = len(info_text) * line_height + margin
        
        # Create transparent background overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (5 + bg_width, 5 + bg_height), (0, 0, 0), -1)
        
        # Apply transparency to the overlay
        alpha = 0.6  # Transparency level (0.0 = fully transparent, 1.0 = fully opaque)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
        # Draw text on top of transparent background
        y_offset = 25
        for text in info_text:
            cv2.putText(frame, text, (10, y_offset),
                        font, font_scale, (255, 255, 255), thickness)
            y_offset += line_height
        
        return frame

    def run(self):
        """Main application loop"""
        # Initialize video capture
        cap = cv2.VideoCapture(self.video_source)
        if not cap.isOpened():
            print(f"Error: Could not open video source {self.video_source}")
            return
        # Reduce buffer size for webcam to minimize latency
        if isinstance(self.video_source, int):
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        # Get frame dimensions for video writer
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        
        # Initialize video writer if recording is enabled
        if self.save_output:
            self._init_video_writer(frame_width, frame_height, source_fps)

        # Create window and set mouse callback
        window_name = "Motion Tracking App"
        cv2.namedWindow(window_name)
        cv2.setMouseCallback(window_name, self.mouse_callback)

        # Print startup information
        print("\n" + "="*50)
        print("MOTION TRACKING APP STARTED")
        print("="*50)
        print(f"MOT Method: {self.mot_method}")
        print(f"SOT Tracker: {self.sot_tracker_type}")
        print(f"YOLO Model: {self.yolo.model.names if hasattr(self.yolo, 'model') else 'Loaded'}")
        print(f"Target Classes: {self.target_classes if self.target_classes else 'All'}")
        if self.save_output:
            print(f"Output Recording: {self.output_path}")
        print("="*50 + "\n")

        # Main processing loop
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            self.frame_count += 1
            self.update_fps()

            # Process frame based on current mode
            if self.sot_mode:
                # Single Object Tracking mode
                if self.sot_tracker is not None:
                    # Update single object tracker
                    success, bbox = self.sot_tracker.update(frame)
                    if success:
                        # Convert bounding box format and draw
                        x, y, w, h = bbox
                        sot_bbox = [x, y, x+w, y+h]
                        frame = self.draw_sot_bbox(frame, sot_bbox)
                    else:
                        # Lost track of object, return to MOT mode
                        self.sot_mode = False
                        self.sot_tracker = None
            else:
                # Multi-Object Tracking mode
                tracks = self.detect_objects(frame)
                if tracks:
                    # Check if user clicked on any bounding box
                    bboxes = [t['bbox'] for t in tracks]
                    clicked_idx = self.check_click_on_bbox(bboxes)
                    if clicked_idx is not None:
                        # Switch to single object tracking for clicked object
                        selected_track = tracks[clicked_idx]
                        self.sot_bbox = selected_track['bbox']
                        self.sot_id = selected_track['id']
                        try:
                            self.sot_tracker = self._init_sot_tracker(frame, self.sot_bbox)
                            self.sot_mode = True
                        except Exception as e:
                            print(f"Failed to init SOT: {e}")
                # Draw all tracked objects
                frame = self.draw_tracks(frame, tracks)

            # Add information overlay
            frame = self.draw_info(frame)
            
            # Write frame to video file if recording
            if self.recording and self.video_writer is not None:
                self.video_writer.write(frame)
            
            # Display frame
            cv2.imshow(window_name, frame)
            
            # Process keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == 27:  # ESC key
                if self.sot_mode:
                    # Return to MOT mode from SOT mode
                    self.sot_mode = False
                    self.sot_tracker = None
            elif key == ord('s'):  # Save frame
                filename = f"frame_{self.frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Saved frame as {filename}")

        # Cleanup resources
        cap.release()
        if self.video_writer is not None:
            self.video_writer.release()
            if self.recording:
                print(f"Video saved: {self.output_path}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Configuration parameters
    VIDEO_SOURCE = "v_test5.mp4"  # Input video source (0 for webcam, or path to video file)
    MOT_METHOD = 'bytetrack'      # Multi-object tracking method ('bytetrack' or 'deepsort')
    SOT_TRACKER = 'csrt'          # Single object tracking algorithm ('csrt', 'kcf', or 'mosse')
    YOLO_MODEL = 'yolov8n.pt'     # YOLO model file
    TARGET_CLASSES = [2,3,4,5,6,7] # COCO dataset class IDs to track (car, motorcycle, bus, truck, etc.)
    CONFIDENCE_THRESHOLD = 0.45   # Minimum confidence for detection
    
    # Video recording settings
    SAVE_OUTPUT = True            # Enable/disable video recording
    OUTPUT_PATH = "tracked_output.mp4"  # Output video file path

    # Create and run tracker
    tracker = SimpleMotionTracker(
        video_source=VIDEO_SOURCE,
        yolo_model=YOLO_MODEL,
        mot_method=MOT_METHOD,
        sot_tracker=SOT_TRACKER,
        target_classes=TARGET_CLASSES,
        confidence_threshold=CONFIDENCE_THRESHOLD,
        save_output=SAVE_OUTPUT,
        output_path=OUTPUT_PATH
    )
    tracker.run()