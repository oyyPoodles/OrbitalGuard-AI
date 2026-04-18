# OrbitalGuard AI 🛰️

**Space Debris Detection, Tracking, and Collision Prevention System**

![Architecture](docs/pipeline_diagram.md)

## The Solution
**OrbitalGuard AI** bridges this critical gap in Space Situational Awareness (SSA) by utilizing a physics-informed AI pipeline to simulate the detection, tracking, risk classification, and autonomous remediation of small orbital debris.

## System Architecture
A modern, decoupled, 3-tier real-time streaming pipeline:

1. **Backend (Python/FastAPI)**: SGP4 propagator, Kalman State Estimation, and logging.
2. **AIML (Intelligence)**: LSTM trajectory correction, XGBoost risk classification, and PPO Reinforcement Avoidance models.
3. **Frontend (Node/React/Three.js)**: NASA-grade 3D hybrid dashboard (60 FPS on 1,500+ objects).

---

## Folder Structure

```
orbitalguard-ai/
├── backend/            # Core Engine & Physics (Port 8000)
├── aiml/               # AI Models, Training Scripts, & Metrics
├── frontend/           # 3D Dashboard & WebSocket Proxy
│   ├── client/         # React Application (Port 5173)
│   └── server/         # Node.js WS Proxy (Port 3001)
├── docs/               # Research Reports, Tables, and Diagrams
└── output/             # (Under aiml/) Generated metrics & presentation plots
```

## How to Run

**1. Start the Backend (AI Engine)**
```bash
cd backend
pip install -r requirements.txt
python app/main.py
```

**2. Start the Proxy Server (Node.js)**
```bash
cd frontend/server
npm install
node index.js
```

**3. Start the Frontend Dashboard (React)**
```bash
cd frontend/client
npm install
npm run dev
```

The visualization dashboard will open at `http://localhost:5173`.

---

## Research Metrics
All mathematical metrics and performance visualizers can be found in `aiml/output/`. To retrain all models and regenerate these metrics from scratch, run:
```bash
python aiml/scripts/train_all_models.py
```
