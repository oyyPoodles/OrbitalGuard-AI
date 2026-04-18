"""
FastAPI Backend — Unified Pipeline Integration (Phase 2 Upgrade)
Additions:
  GET  /api/satellites        → Full satellite list for left sidebar
  POST /api/chat              → Narad AI chatbot endpoint
  GET  /api/object/{id}       → Deep object data (TLE params, risk)
  WS   /ws/live               → Upgraded with altitude + inclination
"""
import asyncio
import json
import sys
import os
import time
import math
import numpy as np
from datetime import datetime, timedelta
from typing import List
from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ─── Path Alignment ───────────────────────────────────────
CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
BACKEND_ROOT = os.path.dirname(CURRENT_DIR)
PROJECT_ROOT = os.path.dirname(BACKEND_ROOT)
AIML_ROOT    = os.path.join(PROJECT_ROOT, 'aiml')

for p in [BACKEND_ROOT, AIML_ROOT]:
    if p not in sys.path:
        sys.path.insert(0, p)

from simulation.environment import OrbitalEnvironment
from tracking.kalman_filter import KalmanFilter6D
from utils.constants import RENDER_SCALE, DETECTION_RADIUS, MAX_OBJECTS

from detection.yolo_simulator import YOLOSimulator
from collision.detector import CollisionDetector
from collision.risk_model import RiskClassifier
from prediction.lstm_model import load_hybrid_model, predict_hybrid_correction
from avoidance.ppo_agent import PPOAvoidanceAgent
from knowledge.orbital_chatbot import NaradChatbot

background_tasks = set()

@asynccontextmanager
async def lifespan(app: FastAPI):
    t1 = asyncio.create_task(physics_loop())
    t2 = asyncio.create_task(data_logger())
    background_tasks.add(t1)
    background_tasks.add(t2)
    yield
    t1.cancel()
    t2.cancel()

app = FastAPI(title="OrbitalGuard AI API v2", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Static Mounts ────────────────────────────────────────
frontend_dir  = os.path.join(PROJECT_ROOT, 'frontend', 'client', 'dist')
analytics_dir = os.path.join(AIML_ROOT, 'output')

if os.path.isdir(frontend_dir):
    app.mount("/ui", StaticFiles(directory=frontend_dir, html=True), name="frontend")
if os.path.isdir(analytics_dir):
    app.mount("/api/analytics", StaticFiles(directory=analytics_dir), name="analytics")

# ─── Global Engines ───────────────────────────────────────
env               = OrbitalEnvironment(max_objects=MAX_OBJECTS)
yolo_sim          = YOLOSimulator(noise_std=0.5)
collision_detector= CollisionDetector(threshold_km=DETECTION_RADIUS)
risk_classifier   = RiskClassifier(model_path=os.path.join(AIML_ROOT, 'models', 'xgb_risk.pkl'))
lstm_model        = load_hybrid_model(os.path.join(AIML_ROOT, 'models', 'hybrid_lstm.pth'))
ppo_agent         = PPOAvoidanceAgent(model_path=os.path.join(AIML_ROOT, 'models', 'ppo_avoidance.zip'))
narad             = NaradChatbot()

# ─── Enrich satellite types from debris catalog ────────────
_catalog_path = os.path.join(AIML_ROOT, 'data', 'debris_catalog.json')
if os.path.exists(_catalog_path):
    with open(_catalog_path, 'r', encoding='utf-8') as _f:
        _catalog = json.load(_f)
    _id_type_map = {}
    for item in _catalog:
        _cid  = str(item.get('id') or '').strip()
        _ctype = str(item.get('type') or '').lower()
        if _cid and _ctype:
            _id_type_map[_cid] = 'starlink' if _ctype == 'starlink' else _ctype
    _enriched = 0
    for sat in env.propagator.satellites:
        sat_id = str(sat['id']).strip()
        if sat_id in _id_type_map:
            mapped = _id_type_map[sat_id]
            sat['type'] = mapped
            _enriched += 1
    print(f"[EnrichTypes] Enriched {_enriched} objects from catalog")
    del _catalog, _id_type_map

# ─── Inject realistic debris distribution ─────────────────
# Real LEO has ~40% debris. Reclassify a portion of payloads
# using a deterministic seed so the map is consistent.
import random as _rnd
_rnd.seed(42)
_sats_list  = env.propagator.satellites
_n_payload  = sum(1 for s in _sats_list if s['type'] == 'payload')
_target_deb = int(len(_sats_list) * 0.38)   # want ~38% debris
_target_rkt = int(len(_sats_list) * 0.05)   # want ~5% rockets
_payload_sats = [s for s in _sats_list if s['type'] == 'payload']
_rnd.shuffle(_payload_sats)

# Assign debris
for i, s in enumerate(_payload_sats[:_target_deb]):
    s['type'] = 'debris'
    s['name'] = f"DEB {s['name'][:10]}"
# Assign rockets (next slice)
for s in _payload_sats[_target_deb:_target_deb + _target_rkt]:
    s['type'] = 'rocket'
    s['name'] = f"R/B {s['name'][:10]}"

_counts = {t: sum(1 for s in _sats_list if s['type'] == t)
           for t in ('payload', 'debris', 'rocket', 'starlink')}
print(f"[TypeDist] {_counts}")

kalman_filters  = {}
object_history  = {}
# ─── State ────────────────────────────────────────────────
current_risks: List[dict] = []
clients: List[WebSocket]  = []
loop_perf_ms              = 0.0

# Pre-build satellite index from env (for /api/satellites)
_satellite_index: List[dict] = []
# Cached per-object alt/inclination (updated every AI tick)
_obj_cache: dict = {}    # id -> {altitude_km, inclination_deg}


def _build_satellite_index():
    """Build a lightweight index of all tracked objects for the left sidebar."""
    global _satellite_index
    objects = env.get_objects()
    _satellite_index = [
        {
            "id":   obj["id"],
            "name": obj["name"].strip(),
            "type": obj["type"],
        }
        for obj in objects
        if not np.any(np.isnan(obj["position"]))
    ]

def _altitude_from_pos(pos: np.ndarray) -> float:
    """Compute approximate altitude (km) from ECI position vector."""
    EARTH_RADIUS_KM = 6371.0
    return max(0.0, float(np.linalg.norm(pos)) - EARTH_RADIUS_KM)

def _inclination_from_vel(pos: np.ndarray, vel: np.ndarray) -> float:
    """Approximate orbital inclination from position/velocity vectors."""
    try:
        h = np.cross(pos, vel)         # angular momentum vector
        inc = math.degrees(math.acos(
            max(-1.0, min(1.0, h[2] / (np.linalg.norm(h) + 1e-10)))
        ))
        return round(inc, 2)
    except Exception:
        return 0.0

def get_valid_objects():
    return [o for o in env.get_objects() if not np.any(np.isnan(o['position']))]

# ─── Physics Loop (decoupled: fast render + slow AI) ──────
AI_TICK_INTERVAL = 15      # Run AI every N ticks (LSTM/KDTree/XGBoost/PPO)
_tick_count = 0

async def physics_loop():
    global current_risks, loop_perf_ms, _tick_count
    _build_satellite_index()

    while True:
        t_start = time.time()
        _tick_count += 1
        env.step(dt_seconds=1.0)
        objects = get_valid_objects()

        # ── YOLO detection (fast) ────────────────────────────
        detections = yolo_sim.detect(objects[:200])    # only first 200 for speed
        for det in detections:
            oid = det['name']
            if oid not in kalman_filters:
                kalman_filters[oid] = KalmanFilter6D()
            kf = kalman_filters[oid]
            kf.predict()
            kf.update(np.concatenate([det['observed_position'], np.zeros(3)]))

        # ── LSTM correction (every AI_TICK_INTERVAL ticks) ──
        if _tick_count % AI_TICK_INTERVAL == 0:
            for obj in objects[:100]:             # cap for performance
                oid = obj['name']
                if oid not in object_history:
                    object_history[oid] = []
                state = (kalman_filters[oid].x.flatten()
                         if oid in kalman_filters
                         else np.concatenate([obj['position'], obj['velocity']]))
                object_history[oid].append(state)
                if len(object_history[oid]) > 10:
                    object_history[oid].pop(0)
                if len(object_history[oid]) == 10:
                    try:
                        obj['position'] += predict_hybrid_correction(
                            lstm_model, np.array(object_history[oid])
                        )
                    except Exception:
                        pass

            # ── KDTree + XGBoost conjunction + risk ─────────
            conjunctions  = collision_detector.detect(objects)
            current_risks = risk_classifier.classify_batch(conjunctions)

            # ── Cache altitude/inclination ────────────────
            for obj in objects:
                _obj_cache[obj['id']] = {
                    'altitude_km':     round(_altitude_from_pos(obj['position']), 1),
                    'inclination_deg': _inclination_from_vel(obj['position'], obj['velocity']),
                }

            # ── PPO avoidance for HIGH risk ──────────────────
            for r in current_risks:
                r['avoidance_maneuver'] = None
                if r.get('risk_level') == 'HIGH':
                    obj1 = next((o for o in objects if o['name'] == r['obj1_id']), None)
                    obj2 = next((o for o in objects if o['name'] == r['obj2_id']), None)
                    if obj1 and obj2:
                        try:
                            maneuver = ppo_agent.compute_avoidance(
                                relative_position=obj2['position'] - obj1['position'],
                                relative_velocity=obj2['velocity'] - obj1['velocity'],
                                own_position=obj1['position'],
                                own_velocity=obj1['velocity'],
                            )
                            r['avoidance_maneuver'] = {
                                "delta_v":   [round(float(v), 3) for v in maneuver['delta_v']],
                                "fuel_cost": round(maneuver['fuel_cost'], 3),
                            }
                        except Exception:
                            pass

        # ── WebSocket broadcast (every tick, lightweight) ──
        if clients:
            payload = []
            for obj in objects[:800]:   # cap render objects for WS performance
                cache  = _obj_cache.get(obj['id'], {})
                payload.append({
                    "id":   obj['id'],
                    "name": obj['name'],
                    "x":    float(obj['position'][0] * RENDER_SCALE),
                    "y":    float(obj['position'][1] * RENDER_SCALE),
                    "z":    float(obj['position'][2] * RENDER_SCALE),
                    "vx":   round(float(obj['velocity'][0]), 4),
                    "vy":   round(float(obj['velocity'][1]), 4),
                    "vz":   round(float(obj['velocity'][2]), 4),
                    "type": obj['type'],
                    "altitude_km":     cache.get('altitude_km', 0),
                    "inclination_deg": cache.get('inclination_deg', 0),
                })

            message = json.dumps({
                "type":    "update",
                "time":    env.get_time().isoformat(),
                "objects": payload,
                "risks": [
                    {
                        "a":           r['obj1_id'],
                        "b":           r['obj2_id'],
                        "distance":    r['distance_km'],
                        "velocity":    r['relative_velocity_kms'],
                        "risk":        r.get('risk_level', 'LOW'),
                        "confidence":  r.get('confidence', 0.0),
                        "probabilities": r.get('probabilities', {}),
                        "maneuver":    r.get('avoidance_maneuver'),
                    }
                    for r in current_risks[:50]
                ],
            })
            for ws in clients[:]:
                try:
                    await ws.send_text(message)
                except Exception:
                    if ws in clients:
                        clients.remove(ws)

        loop_perf_ms = (time.time() - t_start) * 1000.0
        # Maintain ~5Hz: wait 200ms minus time taken (min 10ms)
        await asyncio.sleep(max(0.01, 0.2 - (loop_perf_ms / 1000.0)))


# ─── Logging ──────────────────────────────────────────────
LOG_DIR = os.path.join(BACKEND_ROOT, 'data', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

async def data_logger():
    while True:
        await asyncio.sleep(10)
        objects = get_valid_objects()
        print(f"[Log] Objects: {len(objects)} | Risks: {len(current_risks)} | Loop: {loop_perf_ms:.2f}ms")

# ─── REST Endpoints ───────────────────────────────────────

@app.get("/api/meta")
def get_metadata():
    objects      = get_valid_objects()
    debris_count = len([o for o in objects if o['type'] in ['debris', 'rocket']])
    high_risks   = [r for r in current_risks if r.get('risk_level') == 'HIGH']
    return {
        "total_objects":     len(objects),
        "active_satellites": len(objects) - debris_count,
        "debris_count":      debris_count,
        "high_risk_alerts":  len(high_risks),
        "latency_ms":        round(loop_perf_ms, 2),
        "plots":             [f for f in os.listdir(analytics_dir) if f.endswith('.png')]
                             if os.path.isdir(analytics_dir) else [],
    }


@app.get("/api/satellites")
def get_satellites():
    """
    Returns the full list of tracked objects for the left sidebar.
    Each entry: {id, name, type}
    Sorted alphabetically by name.
    """
    if not _satellite_index:
        _build_satellite_index()
    return sorted(_satellite_index, key=lambda x: x['name'])


@app.get("/api/object/{obj_id}")
def get_object(obj_id: str):
    """
    Return deep data for a single object:
      position, velocity, altitude, inclination, risk history
    """
    objects = get_valid_objects()
    obj = next(
        (o for o in objects if o['id'] == obj_id or o['name'].strip() == obj_id),
        None
    )
    if not obj:
        raise HTTPException(status_code=404, detail=f"Object '{obj_id}' not found")

    alt = _altitude_from_pos(obj['position'])
    inc = _inclination_from_vel(obj['position'], obj['velocity'])

    # Current risk involvement
    obj_risks = [
        r for r in current_risks
        if r.get('obj1_id') == obj['name'] or r.get('obj2_id') == obj['name']
    ]

    return {
        "id":   obj['id'],
        "name": obj['name'].strip(),
        "type": obj['type'],
        "position": {
            "x": round(float(obj['position'][0]), 3),
            "y": round(float(obj['position'][1]), 3),
            "z": round(float(obj['position'][2]), 3),
        },
        "velocity": {
            "x": round(float(obj['velocity'][0]), 4),
            "y": round(float(obj['velocity'][1]), 4),
            "z": round(float(obj['velocity'][2]), 4),
        },
        "altitude_km":      round(alt, 2),
        "inclination_deg":  inc,
        "speed_kms":        round(float(np.linalg.norm(obj['velocity'])), 4),
        "active_risks":     obj_risks,
    }


class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
def chat(req: ChatRequest):
    """Narad AI chatbot endpoint."""
    if not req.query.strip():
        return {"answer": "Please enter a question."}
    answer = narad.ask(req.query.strip())
    return {"answer": answer, "query": req.query}


@app.get("/api/narad")
def narad_get(q: str = ""):
    """Legacy GET endpoint for backward compatibility."""
    if not q.strip():
        return {"answer": "Please enter a question."}
    return {"answer": narad.ask(q.strip())}


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
