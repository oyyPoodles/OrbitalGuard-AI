"""
FastAPI Backend — Unified Pipeline Integration
Pipeline: TLE → SGP4 → YOLO Detection → Kalman → KDTree → XGBoost → WebSocket
"""
import asyncio
import json
import sys
import os
import time
import numpy as np
from datetime import datetime, timedelta
from typing import List

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from simulation.environment import OrbitalEnvironment
from detection.yolo_simulator import YOLOSimulator
from tracking.kalman_filter import KalmanFilter6D
from collision.detector import CollisionDetector
from collision.risk_model import RiskClassifier
from prediction.lstm_model import load_hybrid_model, predict_hybrid_correction
from avoidance.ppo_agent import PPOAvoidanceAgent
from utils.constants import RENDER_SCALE, DETECTION_RADIUS, MAX_OBJECTS

from contextlib import asynccontextmanager

# Store references to background tasks to prevent garbage collection
background_tasks = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    task1 = asyncio.create_task(physics_loop())
    task2 = asyncio.create_task(data_logger())
    background_tasks.add(task1)
    background_tasks.add(task2)
    yield
    # Clean up on shutdown
    task1.cancel()
    task2.cancel()

# ─── App Init ─────────────────────────────────────────────
app = FastAPI(title="OrbitalGuard AI API", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve frontend statically
frontend_dir = os.path.join(os.path.dirname(__file__), '..', 'frontend')
if os.path.isdir(frontend_dir):
    app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="frontend")

# ─── Global Engines ───────────────────────────────────────
env = OrbitalEnvironment(max_objects=MAX_OBJECTS)
yolo_sim = YOLOSimulator(noise_std=0.5)
collision_detector = CollisionDetector(threshold_km=DETECTION_RADIUS)
risk_classifier = RiskClassifier(model_path='models/xgb_risk.pkl')
lstm_model = load_hybrid_model('models/hybrid_lstm.pth')
ppo_agent = PPOAvoidanceAgent(model_path='models/ppo_avoidance.zip')

# Per-object Kalman filters and tracking history for LSTM
kalman_filters = {}
object_history = {}

# ─── State ────────────────────────────────────────────────
current_risks = []
clients: List[WebSocket] = []
loop_perf_ms = 0.0

def get_valid_objects():
    """Get all objects with valid (non-NaN) positions."""
    return [o for o in env.get_objects() if not np.any(np.isnan(o['position']))]


# ─── Background Physics Loop ─────────────────────────────
async def physics_loop():
    global current_risks
    global loop_perf_ms

    while True:
        t_start = time.time()
        
        # 1. Advance simulation by 1 second
        env.step(dt_seconds=1.0)
        objects = get_valid_objects()

        # 2. Simulated Detection (adds noise)
        detections = yolo_sim.detect(objects)

        # 3. Kalman Filtering (smooth each object)
        for det in detections:
            oid = det['name']
            if oid not in kalman_filters:
                kalman_filters[oid] = KalmanFilter6D()
            kf = kalman_filters[oid]
            measurement = np.concatenate([det['observed_position'], np.zeros(3)])
            kf.predict()
            kf.update(measurement)

        # 3.5 Hybrid LSTM Correction Layer
        # Correct the deterministic SGP4 trajectory using LSTM temporal analysis
        for obj in objects:
            oid = obj['name']
            if oid not in object_history:
                object_history[oid] = []
            
            # Use noise-smoothed Kalman state if available
            if oid in kalman_filters:
                state = kalman_filters[oid].state.flatten()
            else:
                state = np.concatenate([obj['position'], obj['velocity']])
                
            object_history[oid].append(state)
            if len(object_history[oid]) > 10:
                object_history[oid].pop(0)
                
            # Apply correction if we have a full sequence (Cold Start Fallback: use pure SGP4 until buffer is full)
            if len(object_history[oid]) == 10:
                try:
                    correction = predict_hybrid_correction(lstm_model, np.array(object_history[oid]))
                    obj['position'] += correction
                except Exception:
                    pass # Ensure robust edge-case handling if prediction fails on corrupted states

        # 4. Collision Detection (KDTree) - Now using corrected trajectories
        conjunctions = collision_detector.detect(objects)

        # 5. Risk Classification (XGBoost)
        current_risks = risk_classifier.classify_batch(conjunctions)

        # 6. Autonomous Avoidance (PPO RL) for HIGH risks
        for r in current_risks:
            r['avoidance_maneuver'] = None
            if r.get('risk_level') == 'HIGH':
                obj1 = next((o for o in objects if o['name'] == r['obj1_id']), None)
                obj2 = next((o for o in objects if o['name'] == r['obj2_id']), None)
                if obj1 and obj2:
                    maneuver = ppo_agent.compute_avoidance(
                        relative_position=obj2['position'] - obj1['position'],
                        relative_velocity=obj2['velocity'] - obj1['velocity'],
                        own_position=obj1['position'],
                        own_velocity=obj1['velocity']
                    )
                    r['avoidance_maneuver'] = {
                        "delta_v": [round(float(v), 3) for v in maneuver['delta_v']],
                        "fuel_cost": round(maneuver['fuel_cost'], 3)
                    }

        # 7. Broadcast to WebSocket clients
        if clients:
            payload = []
            for obj in objects[:MAX_OBJECTS]:
                payload.append({
                    "id": obj['id'],
                    "name": obj['name'],
                    "x": float(obj['position'][0] * RENDER_SCALE),
                    "y": float(obj['position'][1] * RENDER_SCALE),
                    "z": float(obj['position'][2] * RENDER_SCALE),
                    "type": obj['type']
                })

            risk_payload = []
            for r in current_risks[:50]:
                entry = {
                    "a": r['obj1_id'], "b": r['obj2_id'],
                    "distance": r['distance_km'], "risk": r.get('risk_level', 'LOW')
                }
                if r.get('avoidance_maneuver'):
                    entry["maneuver"] = r['avoidance_maneuver']
                risk_payload.append(entry)

            message = json.dumps({
                "type": "update",
                "time": env.get_time().isoformat(),
                "objects": payload,
                "risks": risk_payload
            })

            disconnected = []
            for ws in clients:
                try:
                    await ws.send_text(message)
                except:
                    disconnected.append(ws)
            for ws in disconnected:
                clients.remove(ws)

        # Performance timing
        loop_perf_ms = (time.time() - t_start) * 1000.0

        # Dynamic sleep to target 10 Hz (100ms) cycle time
        sleep_time = max(0.01, 0.1 - (loop_perf_ms / 1000.0))
        await asyncio.sleep(sleep_time)


# ─── Data Logging ─────────────────────────────────────────
DATASET_DIR = os.path.join(os.path.dirname(__file__), '..', 'dataset')
os.makedirs(DATASET_DIR, exist_ok=True)

async def data_logger():
    """Non-blocking logger: writes to dataset/ every 10 seconds."""
    while True:
        await asyncio.sleep(10)
        ts = env.get_time().isoformat()
        objects = get_valid_objects()

        try:
            # Objects log
            obj_entry = {"timestamp": ts, "count": len(objects), "objects": [
                {"name": o['name'], "position": o['position'].tolist(), "velocity": o['velocity'].tolist(), "type": o['type']}
                for o in objects[:200]  # sample for file size
            ]}
            with open(os.path.join(DATASET_DIR, 'objects_log.json'), 'a') as f:
                f.write(json.dumps(obj_entry) + '\n')

            # Collision log
            conj = collision_detector.detect(objects)
            high_risks = [r for r in current_risks if r.get('risk_level') == 'HIGH']
            avg_dist = sum([c['distance_km'] for c in conj]) / len(conj) if conj else 0.0

            col_entry = {
                "timestamp": ts, 
                "summary": {
                    "total_objects": len(objects),
                    "total_collisions_tracked": len(conj),
                    "high_risk_count": len(high_risks),
                    "average_distance_km": round(avg_dist, 2)
                },
                "collisions": [
                    {"object1": c['obj1_id'], "object2": c['obj2_id'], "distance": c['distance_km']}
                    for c in conj[:50]
                ]
            }
            with open(os.path.join(DATASET_DIR, 'collision_log.json'), 'a') as f:
                f.write(json.dumps(col_entry) + '\n')

            # Risk log
            risk_entry = {"timestamp": ts, "risks": [
                {"object1": r['obj1_id'], "object2": r['obj2_id'], "risk_level": r.get('risk_level', 'LOW')}
                for r in current_risks[:50]
            ]}
            with open(os.path.join(DATASET_DIR, 'risk_log.json'), 'a') as f:
                f.write(json.dumps(risk_entry) + '\n')

            print(f"[Log] System Healthy | Objects: {len(objects)} | Collisions: {len(conj)} | Risks: {len(current_risks)} | Loop Time: {loop_perf_ms:.2f} ms")
        except Exception as e:
            print(f"[Warning] Logging error: {e}")





# ─── REST Endpoints ───────────────────────────────────────
@app.get("/objects")
def get_objects_endpoint():
    objects = get_valid_objects()
    return [{
        "id": o['id'], "name": o['name'],
        "x": float(o['position'][0]),
        "y": float(o['position'][1]),
        "z": float(o['position'][2]),
        "type": o['type']
    } for o in objects[:MAX_OBJECTS]]


@app.get("/risks")
def get_risks_endpoint():
    return current_risks[:100]


# ─── WebSocket ────────────────────────────────────────────
@app.websocket("/ws/live")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    clients.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        if websocket in clients:
            clients.remove(websocket)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
