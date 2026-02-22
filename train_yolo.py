import os
import yaml
from ultralytics import YOLO

def train_yolo_model():
    """
    Trains a YOLOv8 Nano model on the space debris imagery dataset
    to detect real debris bounding boxes.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(base_dir, "data", "debris_images")
    yaml_path = os.path.join(data_dir, "data.yaml")
    
    # Rewrite data.yaml to enforce valid absolute pathing
    if os.path.exists(yaml_path):
        with open(yaml_path, 'r') as f:
            data_config = yaml.safe_load(f)
            
        data_config['path'] = data_dir.replace("\\", "/")
        data_config['train'] = 'train/images'
        data_config['val'] = 'valid/images'
        data_config['test'] = 'test/images'
        
        with open(yaml_path, 'w') as f:
            yaml.safe_dump(data_config, f)
        print(f"✅ Fixed absolute pathing in {yaml_path}")
    else:
        print(f"❌ Could not find {yaml_path}. Is the dataset downloaded?")
        return

    # Initialize a pretrained YOLOv8 model (Nano for fast training)
    print("\n🚀 Initializing YOLOv8 Nano model...")
    model = YOLO("yolov8n.pt") 
    
    # Train the model
    print("🧠 Commencing Training Phase (10 epochs for demonstration)...")
    model.train(
        data=yaml_path,
        epochs=3, 
        imgsz=640,
        batch=16,
        project=os.path.join(base_dir, "models"),
        name="yolov8_debris",
        exist_ok=True
    )
    
    print("\n📊 Evaluating YOLOv8 metrics on the validation dataset...")
    metrics = model.val()
    
    print("\n" + "="*50)
    print("🏆 YOLOv8 Debris Detection Training Complete!")
    print("="*50)
    print(f"🎯 mAP50-95: {metrics.box.map:.4f}")
    print(f"🎯   mAP50: {metrics.box.map50:.4f}")
    print(f"🔎 Precision: {metrics.box.mp:.4f}")
    print(f"🔎    Recall: {metrics.box.mr:.4f}")
    print("="*50)
    print(f"📁 Optimal weights saved to: models/yolov8_debris/weights/best.pt")

if __name__ == "__main__":
    train_yolo_model()
