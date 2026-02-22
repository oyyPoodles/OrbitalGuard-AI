import os
import cv2
import numpy as np

# Require ultralytics strictly now
from ultralytics import YOLO

class DebrisDetector:
    """
    YOLOv8 Wrapper for identifying space debris in optical feeds.
    Strictly requires a trained .pt weight file.
    """
    def __init__(self, model_path: str = "models/yolov8_debris/weights/best.pt"):
        # We explicitly rely on the path we wrote in train_yolo.py
        self.model_path = model_path
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"[Detector] Critical: Trained Debris YOLOv8 weights not found at {model_path}. Run train_yolo.py first.")
            
        print(f"[Detector] Loading Research-Grade YOLOv8 weights from {model_path}...")
        self.model = YOLO(self.model_path)
            
    def detect(self, image_bgr: np.ndarray) -> dict:
        """
        Runs true inference on an image frame using trained YOLO.
        Outputs a standardized dictionary of bounding boxes and confidences.
        """
        results = self.model(image_bgr, verbose=False)[0]
        boxes = results.boxes.xyxy.cpu().numpy().tolist()
        confs = results.boxes.conf.cpu().numpy().tolist()
        classes = [results.names[int(c)] for c in results.boxes.cls.cpu().numpy()]
        
        return {
            "boxes": boxes,
            "confidences": confs,
            "classes": classes
        }
            
    def draw_predictions(self, image_bgr: np.ndarray, detections: dict) -> np.ndarray:
        """Helper to overlay the bounding boxes back onto the image."""
        img_copy = image_bgr.copy()
        for idx, box in enumerate(detections['boxes']):
            x1, y1, x2, y2 = map(int, box)
            conf = detections['confidences'][idx]
            cls = detections['classes'][idx]
            
            # Draw Box
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
            # Draw Label
            label = f"{cls} {conf:.2f}"
            cv2.putText(img_copy, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
        return img_copy

if __name__ == "__main__":
    # Test the module throws error before training finishes
    try:
        detector = DebrisDetector()
    except Exception as e:
        print(e)
