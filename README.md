# OrbitalGuard AI 🛰️

**Space Debris Detection, Tracking, and Collision Prevention System**

![Architecture](docs/architecture.png)

## The Problem (NASA Challenge Alignment)
Decades of space exploration have left Low Earth Orbit (LEO) littered with millions of untracked debris fragments (1mm–10cm) traveling at speeds exceeding 7–8 km/s. Even microscopic particles can critically damage operational satellites and space infrastructure. Traditional radar systems lack the resolution and cost-efficiency to comprehensively track these micro-threats.

## The Solution
**OrbitalGuard AI** bridges this critical gap in Space Situational Awareness (SSA) by utilizing a physics-informed AI pipeline to simulate the detection, tracking, risk classification, and autonomous remediation of small orbital debris.

## Core Features (NASA Workflow)

1. **Detect (Physics + Vision Simulation)**
   - **SGP4 Propagation**: Accurately computes orbital state vectors from raw TLE data.
   - **YOLO Simulation**: Simulates the detection of sub-10cm debris, identifying objects and introducing realistic sensor noise.

2. **Track & Predict (ML Operations)**
   - **Kalman Filtering**: Smooths real-time detection noise to acquire stable state-estimation tracking.
   - **LSTM Prediction**: Projects future trajectories based on temporal tracking sequences.

3. **Remediate (Collision & Avoidance)**
   - **KDTree Optimization**: Achieves sub-second collision detection across thousands of orbiting objects.
   - **XGBoost Risk Classification**: Classifies conjunction events into High, Medium, or Low risk tiers based on proximity, velocity, and object mass.
   - **PPO Avoidance**: Uses Reinforcement Learning to calculate optimal $\Delta V$ maneuvers to avoid high-risk collisions.

## System Architecture
A modern, decoupled, 3-tier real-time streaming pipeline:

1. **AI Engine (Python/FastAPI)** → SGP4, ML inferences, KDTree, websocket broadcasting.
2. **Proxy Server (Node.js)** → Connection multiplexing, state caching, broadcast optimization.
3. **NASA-Grade Dashboard (React + Three.js)** → 3D hybrid rendering engine (60 FPS on 1,500+ objects), real-time risk event tabular readouts, and `dataset` JSON exporting.

---

## Folder Structure

```
orbitalguard/
├── backend/            # FastAPI, AI Engine, Physics models
├── frontend/           # React + Three.js UI
├── server/             # Node.js WebSocket Proxy
├── dataset/            # Data logging (JSON risk outputs)
├── models/             # PyTorch & XGBoost models
├── data/               # TLE Datasets, raw configs
└── docs/               # Research reports and architecture diagrams
```

## How to Run

*Requires Python 3.10+, Node.js 18+*

**1. Start the Backend (AI Engine)**
```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

**2. Start the Proxy Server (Node.js)**
```bash
cd server
npm install
node index.js
```

**3. Start the Frontend Dashboard (React)**
```bash
cd frontend
npm install
npm run dev
```

The visualization dashboard will open at `http://localhost:5173`.
