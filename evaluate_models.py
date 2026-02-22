import os
import sys
import numpy as np
import cv2

# Set PATH for modules
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# Modules
from detection.yolo_detector import DebrisDetector
from prediction.kalman_filter import OrbitalKalmanFilter
from prediction.lstm_predictor import TrajectoryPredictor
from collision.risk_predictor import CollisionRiskModel
from avoidance.rl_agent import AvoidanceAgent

def generate_test_image() -> np.ndarray:
    """Gets an image from the debris dataset to test YOLO inference."""
    # Build a simple synthetic starry background with a "debris" block 
    # just in case the dataset images are not yet ready/downloaded properly.
    # A true test evaluates the pipeline's handling of image bounds.
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    for _ in range(100):
        x, y = np.random.randint(0, 640, 2)
        cv2.circle(img, (x, y), 1, (255, 255, 255), -1)
        
    # Draw debris (dark grey rectangle)
    cv2.rectangle(img, (300, 300), (340, 340), (100, 100, 100), -1)
    return img

def evaluate_integration_pipeline():
    print("\n" + "="*50)
    print("🛰️  SYSTEM END-TO-END VERIFICATION: TRUE INFERENCE MODE")
    print("="*50)
    
    # 1. Initialize strictly trained modules
    print("Loading Weights... (If any module fails, it means .pt / .pth / .zip is missing)")
    try:
        yolo = DebrisDetector()
        print("✅ YOLOv8 Loaded.")
        
        kalman = OrbitalKalmanFilter()
        print("✅ Kalman Filter Initialized.")
        
        lstm = TrajectoryPredictor()
        print("✅ Sequence LSTM Loaded.")
        
        xgb = CollisionRiskModel()
        print("✅ XGBoost Risk Classifier Loaded.")
        
        rl = AvoidanceAgent()
        print("✅ PPO RL Agent Loaded.")
    except Exception as e:
        print(f"❌ INTEGRATION FAILURE: {e}")
        return
        
    print("\n" + "-"*50)
    print("1. CV DETECTION (YOLOv8)")
    print("-"*50)
    test_img = generate_test_image()
    detections = yolo.detect(test_img)
    print(f"Detected {len(detections['boxes'])} objects. Boxes: {detections['boxes']}")
    
    # Simulate converting bounding boxes to rough spatial coordinates (X, Y, Z)
    # over 10 frames to build an LSTM sequence buffer.
    print("\n" + "-"*50)
    print("2. TRACKING & PREDICTION (KALMAN + LSTM)")
    print("-"*50)
    
    raw_observations = np.array([
        [100.0, 50.0, 10.0] + np.random.normal(0, 2.0, 3) for _ in range(10)
    ])
    
    filtered_seq = []
    print("Applying 6D Kalman Filter to noisy observations...")
    for obs in raw_observations:
        kalman.predict()
        f_obs = kalman.update(obs)
        filtered_seq.append(f_obs)
        
    filtered_seq = np.array(filtered_seq)
    
    # Send historical sequence to LSTM
    next_pos_pred = lstm.predict_next_position(filtered_seq)
    print(f"Noisy Head: {np.round(raw_observations[-1], 2)}")
    print(f"Filtered Head: {np.round(filtered_seq[-1], 2)}")
    print(f"LSTM Forecast (t+1): {np.round(next_pos_pred, 2)}")
    
    print("\n" + "-"*50)
    print("3. COLLISION RISK ASSESSMENT (XGBoost)")
    print("-"*50)
    # Satellite is at 0,0,0. 
    miss_dist = np.linalg.norm(next_pos_pred) 
    # Simulate high relative velocity
    rel_vel = 14.5 
    
    risk = xgb.predict_risk(miss_dist, rel_vel)
    color = "🛑" if risk['class'] == 'HIGH' else "⚠️" if risk['class'] == 'MEDIUM' else "✅"
    print(f"Miss Distance: {miss_dist:.2f} km | Relative Velocity: {rel_vel} km/s")
    print(f"Classification: {color} {risk['class']} (Confidence: {risk['score']:.2%})")
    
    print("\n" + "-"*50)
    print("4. AUTONOMOUS AVOIDANCE MANEUVER (PPO)")
    print("-"*50)
    if risk['class'] in ['HIGH', 'MEDIUM']:
        # Build state [sat_pos, debris_pos, sat_vel]
        sat_pos = np.zeros(3)
        sat_vel = np.array([7.0, 0, 0])
        debris_pos = next_pos_pred
        
        state = np.concatenate([sat_pos, debris_pos, sat_vel]).astype(np.float32)
        maneuver = rl.suggest_maneuver(state)
        
        print(f"Agent executing evasive Delta-V burn:")
        print(f"  ΔV_x: {maneuver[0]:.4f} km/s")
        print(f"  ΔV_y: {maneuver[1]:.4f} km/s")
        print(f"  ΔV_z: {maneuver[2]:.4f} km/s")
        print(f"  Total Effort: {np.linalg.norm(maneuver):.4f} km/s")
    else:
        print("Trajectory safe. Thrusters on standby.")
        
    print("\n" + "="*50)
    print("SUCCESS: ALL SYSTEM MODULES EXECUTED IN PIPELINE")
    print("="*50)

if __name__ == "__main__":
    evaluate_integration_pipeline()
