<div align="center">

```
 ██████╗ ██████╗ ██████╗ ██╗████████╗ █████╗ ██╗      ██████╗ ██╗   ██╗ █████╗ ██████╗ ██████╗
██╔═══██╗██╔══██╗██╔══██╗██║╚══██╔══╝██╔══██╗██║     ██╔════╝ ██║   ██║██╔══██╗██╔══██╗██╔══██╗
██║   ██║██████╔╝██████╔╝██║   ██║   ███████║██║     ██║  ███╗██║   ██║███████║██████╔╝██║  ██║
██║   ██║██╔══██╗██╔══██╗██║   ██║   ██╔══██║██║     ██║   ██║██║   ██║██╔══██║██╔══██╗██║  ██║
╚██████╔╝██║  ██║██████╔╝██║   ██║   ██║  ██║███████╗╚██████╔╝╚██████╔╝██║  ██║██║  ██║██████╔╝
 ╚═════╝ ╚═╝  ╚═╝╚═════╝ ╚═╝   ╚═╝   ╚═╝  ╚═╝╚══════╝ ╚═════╝  ╚═════╝ ╚═╝  ╚═╝╚═╝  ╚═╝╚═════╝
```

### 🛰️ &nbsp; *Defending Low Earth Orbit with Physics-Informed AI* &nbsp; 🛰️

<br/>

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Node.js](https://img.shields.io/badge/Node.js-18+-339933?style=for-the-badge&logo=nodedotjs&logoColor=white)](https://nodejs.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://reactjs.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Three.js](https://img.shields.io/badge/Three.js-r160-000000?style=for-the-badge&logo=threedotjs&logoColor=white)](https://threejs.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)](https://pytorch.org)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-006400?style=for-the-badge)](https://xgboost.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-F7DF1E?style=for-the-badge)](LICENSE)

<br/>

> *"A single 1cm bolt travelling at 7 km/s carries the kinetic energy of a hand grenade."*
> — NASA Orbital Debris Program Office

<br/>

**OrbitalGuard AI** is a full-stack, physics-informed AI system for real-time detection, tracking,
risk classification, and autonomous collision avoidance of sub-10cm orbital debris —
the most dangerous and least-tracked threat class in Low Earth Orbit.

<br/>

[![NASA Challenge](https://img.shields.io/badge/🚀_NASA-Space_Apps_Challenge-0B3D91?style=for-the-badge)](https://www.spaceappschallenge.org)
[![Status](https://img.shields.io/badge/Status-Active_Development-brightgreen?style=for-the-badge)]()
[![LEO Coverage](https://img.shields.io/badge/Coverage-Low_Earth_Orbit-blueviolet?style=for-the-badge)]()

</div>

<br/>

---

<div align="center">

## 📌 &nbsp; Navigation

[🌍 The Problem](#-the-problem) &nbsp;·&nbsp;
[✨ Features](#-core-features) &nbsp;·&nbsp;
[🏗️ Architecture](#️-system-architecture) &nbsp;·&nbsp;
[🔄 Workflow](#-end-to-end-workflow) &nbsp;·&nbsp;
[🗂️ Structure](#️-folder-structure) &nbsp;·&nbsp;
[⚡ Quick Start](#-quick-start) &nbsp;·&nbsp;
[🔬 Tech Stack](#-tech-stack)

</div>

---

<br/>

## 🌍 &nbsp; The Problem

<div align="center">

*The orbital debris crisis is invisible, silent, and accelerating.*

</div>

<br/>

Decades of space launches have turned Low Earth Orbit into a **high-speed debris field**. Fragments from rocket bodies, defunct satellites, and collision events now travel at velocities that make even microscopic particles lethal to active spacecraft.

<br/>

<div align="center">

| &nbsp; | Threat | Scale |
|:---:|:---|:---|
| 🔴 | Untracked debris fragments in LEO | **Millions** |
| 🟠 | Average fragment collision velocity | **7 – 8 km/s** |
| 🟡 | Minimum size trackable by ground radar | **~10 cm** |
| 🔵 | Fragments below radar threshold | **Tens of millions** |
| 🟢 | OrbitalGuard AI detection target | **< 1 mm** |

</div>

<br/>

> 💡 **The Gap:** Ground-based radar tracks objects ≥ 10 cm. Objects between **1 mm and 10 cm** — the most numerous debris class — are essentially invisible to current monitoring systems. OrbitalGuard AI is designed to close this gap entirely.

<br/>

---

<br/>

## ✨ &nbsp; Core Features

<br/>

### &nbsp; 🔭 &nbsp; Phase 1 · Detect &nbsp; `Physics + Vision Simulation`

<div align="center">

| Component | Technology | What It Does |
|:---|:---:|:---|
| **Orbital Propagation** | SGP4 | Converts raw TLE data into precise 6D state vectors (position + velocity) |
| **Debris Detection** | YOLOv8 | Simulates sub-10cm object detection with realistic sensor noise injection |

</div>

<br/>

### &nbsp; 📡 &nbsp; Phase 2 · Track & Predict &nbsp; `ML Operations`

<div align="center">

| Component | Technology | What It Does |
|:---|:---:|:---|
| **State Estimation** | Kalman Filter | Smooths noisy real-time detections into stable trajectory estimates |
| **Trajectory Forecasting** | PyTorch LSTM | Predicts future orbital positions over configurable time horizons |

</div>

<br/>

### &nbsp; 🚀 &nbsp; Phase 3 · Remediate &nbsp; `Collision Avoidance`

<div align="center">

| Component | Technology | What It Does |
|:---|:---:|:---|
| **Proximity Search** | SciPy KDTree | Sub-second conjunction detection across **1,500+** simultaneous objects |
| **Risk Classification** | XGBoost | Tiers conjunctions as 🔴 HIGH / 🟡 MEDIUM / 🟢 LOW with explainable features |
| **ΔV Planning** | PPO (RL) | Computes fuel-optimal avoidance maneuvers via reinforcement learning |

</div>

<br/>

---

<br/>

## 🏗️ &nbsp; System Architecture

<div align="center">

*A modern, decoupled **3-tier real-time streaming pipeline** — AI engine to 3D dashboard.*

</div>

<br/>

```mermaid
flowchart LR

    %% ── DATA SOURCES ──
    subgraph SRC ["  📡  Data Sources  "]
        A1["🛰️ Satellite Imagery"]
        A2["📻 Radar Systems"]
        A3["🔭 Optical Telescopes"]
        A4["📈 Telemetry Streams"]
        A5["🖥️ Simulation Engine"]
    end

    %% ── INGESTION ──
    B["📥 Data Ingestion & Fusion
    ─────────────────────────
    Streaming + Batch Processing"]

    A1 & A2 & A3 & A4 & A5 --> B

    %% ── PREPROCESSING ──
    C["⚙️ Preprocessing & Feature Engineering
    ─────────────────────────────────────
    Noise Reduction · Normalization · Fusion"]
    B --> C

    %% ── DETECT ──
    subgraph DETECT ["  🔍  Detect & Characterize  "]
        D1["YOLOv8 Detection
        ─────────────────
        Sub-10cm Objects"]
        D2["Object Characterization
        ─────────────────────
        Size · Shape · Velocity"]
        D1 --> D2
    end
    C --> D1

    %% ── TRACK ──
    subgraph TRACK ["  📊  Tracking System  "]
        E1["Multi-Object Tracking
        ──────────────────────
        Kalman Filter / DeepSORT"]
        E2["Trajectory Estimation
        ─────────────────────
        SGP4 Orbital Mechanics"]
        E3[("🗄️ Trajectory DB
        Real-Time")]
        E1 --> E2 --> E3
    end
    D2 --> E1

    %% ── PREDICT ──
    subgraph PRED ["  🧠  Collision Prediction  "]
        F1["Relative Motion Analysis
        ─────────────────────────
        KDTree Conjunction Search"]
        F2["XGBoost Risk Model
        ──────────────────
        HIGH · MEDIUM · LOW"]
        F1 --> F2
    end
    E3 --> F1

    %% ── DECISION ──
    G{{"⚠️ Risk
    Above
    Threshold?"}}
    F2 --> G

    %% ── REMEDIATION ──
    subgraph REM ["  🚀  Autonomous Remediation  "]
        H1["PPO RL Agent
        ─────────────
        ΔV Computation"]
        H2["Trajectory
        Optimization"]
        H3["Strategy Selection
        ───────────────────
        Deorbit · Laser · Capture"]
        H1 --> H2 --> H3
    end
    G -- "🔴 YES" --> H1
    G -- "🟢 NO"  --> I["🔄 Continue
    Monitoring"]

    %% ── OUTPUT ──
    subgraph OUT ["  🖥️  Command & Control  "]
        J1["🛰️ Satellite Commands"]
        J2["🚨 Alert System"]
        J3["📊 Mission Dashboard
        ─────────────────────
        React + Three.js · 60 FPS"]
    end
    H3 --> J1
    F2 --> J2
    E3 & H1 --> J3

    %% ── STORAGE ──
    subgraph STORE ["  🗃️  Persistent Storage  "]
        K1[("📦 Dataset Repo")]
        K2[("🧠 Model Weights")]
        K3[("📋 Logs & Metrics")]
    end
    B --> K1
    D1 --> K2
    F2 --> K3
```

<br/>

---

<br/>

## 🔄 &nbsp; End-to-End Workflow

<div align="center">

*Full sequence from TLE upload to satellite maneuver command — 5 phases, fully automated.*

</div>

<br/>

```mermaid
sequenceDiagram
    autonumber

    participant U  as 👤 Mission Control
    participant GW as 🌐 API Gateway
    participant IN as 📥 Ingestion Layer
    participant PP as ⚙️ Preprocessing
    participant YO as 🔍 YOLO Detector
    participant KF as 📊 Kalman Tracker
    participant LS as 🧠 LSTM Predictor
    participant XG as ⚖️ XGBoost Classifier
    participant RL as 🤖 PPO Agent
    participant DB as 🗃️ Data Store
    participant UI as 🖥️ React Dashboard

    rect rgb(13, 27, 62)
        Note over U,GW: ━━━━━━━━━  PHASE 1 · DATA INGESTION  ━━━━━━━━━
        U  ->> GW : Upload TLE dataset + sensor config
        GW ->> IN : POST /ingest  [stream open]
        IN ->> PP : Raw TLE + multi-source sensor streams
        PP ->> PP : Clean · normalize · feature engineer
        PP -->> DB: Persist preprocessed records
        PP -->> UI: Ingestion status update
    end

    rect rgb(10, 50, 35)
        Note over PP,KF: ━━━━━━━━━  PHASE 2 · DETECT & TRACK  ━━━━━━━━━
        PP  ->> YO : Processed frame batches
        YO  ->> YO : SGP4 propagation → YOLO inference
        Note right of YO: Realistic sensor noise injected
        YO -->> KF : Detected objects + bounding metadata
        KF  ->> KF : Multi-object Kalman smoothing (EKF)
        KF -->> LS : Stable trajectory state sequences
        KF -->> UI : Live 3D position stream
    end

    rect rgb(55, 28, 10)
        Note over LS,XG: ━━━━━━━━━  PHASE 3 · PREDICT & CLASSIFY  ━━━━━━━━━
        LS  ->> LS : LSTM forward pass (N time steps)
        LS -->> XG : Predicted future state vectors
        XG  ->> XG : KDTree conjunction analysis
        XG  ->> XG : Feature extraction → XGBoost inference
        XG -->> GW : Risk tier: 🔴 HIGH · 🟡 MEDIUM · 🟢 LOW
        XG -->> DB : Persist predictions + confidence scores
        XG -->> UI : Risk event table update
    end

    alt 🔴 HIGH RISK — Miss distance < 1 km
        rect rgb(65, 10, 10)
            Note over XG,RL: ━━━━━━━━━  PHASE 4A · AUTONOMOUS AVOIDANCE  ━━━━━━━━━
            GW  ->> RL : Trigger PPO agent [state vector + constraints]
            RL  ->> RL : Simulate avoidance trajectories
            RL  ->> RL : Optimise for min ΔV + fuel cost
            RL -->> GW : Optimal avoidance trajectory
            GW -->> U  : 🚨 CRITICAL ALERT + maneuver command
            GW -->> UI : Push high-risk collision event
            UI -->> U  : 3D maneuver overlay rendered
        end
    else 🟡 MEDIUM RISK — Miss distance 1–5 km
        rect rgb(55, 45, 5)
            Note over GW,UI: ━━━━━━━━━  PHASE 4B · ENHANCED MONITORING  ━━━━━━━━━
            GW -->> UI : Add to watchlist · increase poll rate
            GW -->> U  : 📋 Advisory notification dispatched
        end
    else 🟢 LOW RISK — Miss distance > 5 km
        rect rgb(10, 55, 20)
            Note over GW,UI: ━━━━━━━━━  PHASE 4C · PASSIVE MONITORING  ━━━━━━━━━
            GW -->> UI : Standard telemetry heartbeat
        end
    end

    rect rgb(20, 18, 55)
        Note over U,UI: ━━━━━━━━━  PHASE 5 · VISUALISE & EXPORT  ━━━━━━━━━
        KF -->> UI : Real-time 3D orbital positions (60 FPS)
        XG -->> UI : Live risk event tabular readout
        RL -->> UI : Avoidance maneuver trajectory overlay
        U  ->> UI  : Request dataset export
        UI -->> U  : 📦 risk_dataset.json  [conjunction log]
    end
```

<br/>

---

<br/>

## 🧠 &nbsp; ML Pipeline at a Glance

<br/>

```
  ┌─────────────┐
  │  TLE  Data  │  ← Two-Line Element sets (NORAD / Space-Track)
  └──────┬──────┘
         │  SGP4 propagation
         ▼
  ┌─────────────────────┐
  │  YOLO v8 Detection  │  ← Sub-10cm debris with sensor noise simulation
  └──────────┬──────────┘
             │  Bounding boxes + confidence scores
             ▼
  ┌────────────────────────┐
  │  Kalman Filter Tracker  │  ← Extended Kalman Filter (EKF) smoothing
  └────────────┬────────────┘
               │  Stable trajectory sequences
               ▼
  ┌─────────────────────┐
  │  LSTM  Forecaster   │  ← N-step future position prediction
  └──────────┬──────────┘
             │  Predicted state vectors
             ▼
  ┌──────────────────────────┐
  │  KDTree Conjunction Search│  ← Sub-second proximity queries
  └─────────────┬────────────┘
                │  Conjunction candidates
                ▼
  ┌─────────────────────────┐
  │  XGBoost Risk Classifier │
  └──────┬──────┬──────┬────┘
         │      │      │
         ▼      ▼      ▼
      🔴 HIGH  🟡 MED  🟢 LOW
         │
         ▼
  ┌─────────────────┐
  │  PPO RL  Agent  │  ← Stable-Baselines3 · fuel-optimal ΔV
  └────────┬────────┘
           │  Avoidance trajectory
           ▼
  ┌──────────────────────┐
  │  Satellite  Command  │  ← Uplinked via mission control interface
  └──────────────────────┘
```

<br/>

---

<br/>

## 📊 &nbsp; Risk Classification Tiers

<br/>

<div align="center">

| Tier | Probability of Collision | Miss Distance | Automated Response |
|:---:|:---|:---:|:---|
| 🔴 &nbsp; **HIGH** | > 1 in 1,000 | < 1 km | PPO maneuver computed + 🚨 immediate alert issued |
| 🟡 &nbsp; **MEDIUM** | 1 in 1,000 – 10,000 | 1 – 5 km | Added to watchlist + 📋 advisory dispatched |
| 🟢 &nbsp; **LOW** | < 1 in 10,000 | > 5 km | Passive monitoring continues |

</div>

<br/>

---

<br/>

## 🗂️ &nbsp; Folder Structure

<br/>

```
orbitalguard/
│
├── 🐍  backend/                   # Python · FastAPI · AI Engine
│   ├── app/
│   │   ├── main.py                #  Entrypoint + WebSocket broadcaster
│   │   ├── sgp4_engine.py         #  Orbital propagation  (TLE → state vectors)
│   │   ├── yolo_sim.py            #  YOLO debris detection simulation
│   │   ├── kalman.py              #  Extended Kalman Filter tracker
│   │   ├── lstm_model.py          #  Trajectory prediction model
│   │   ├── kdtree.py              #  KDTree conjunction queries
│   │   ├── xgboost_risk.py        #  Risk classification pipeline
│   │   └── ppo_agent.py           #  PPO reinforcement learning agent
│   └── requirements.txt
│
├── ⚛️   frontend/                  # React 18 · Three.js · Vite
│   ├── src/
│   │   ├── components/
│   │   │   ├── Globe3D.tsx         #  Three.js 3D orbital visualization
│   │   │   ├── RiskTable.tsx       #  Real-time conjunction event table
│   │   │   └── ManeuverOverlay.tsx #  ΔV avoidance trajectory renderer
│   │   └── App.tsx
│   └── package.json
│
├── 🟩  server/                    # Node.js WebSocket Proxy
│   ├── index.js                   #  Connection multiplexing + state caching
│   └── package.json
│
├── 📦  dataset/                   # JSON risk event exports
├── 🧠  models/                    # PyTorch (.pt) & XGBoost (.ubj) weights
├── 📂  data/                      # TLE datasets · sensor configs
└── 📄  docs/                      # Research reports · architecture diagrams
```

<br/>

---

<br/>

## ⚡ &nbsp; Quick Start

<div align="center">

> **Prerequisites** &nbsp;·&nbsp; ![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square) &nbsp; ![Node](https://img.shields.io/badge/Node.js-18+-green?style=flat-square) &nbsp; ![npm](https://img.shields.io/badge/npm-9+-red?style=flat-square)

</div>

<br/>

### &nbsp; `Step 1` &nbsp; — &nbsp; Start the AI Engine

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

> 🤖 &nbsp; AI engine live at **`http://localhost:8000`** · WebSocket at **`ws://localhost:8000/ws`**

<br/>

### &nbsp; `Step 2` &nbsp; — &nbsp; Start the Proxy Server

```bash
cd server
npm install
node index.js
```

> 🔀 &nbsp; Proxy handling connection multiplexing on **`port 3001`**

<br/>

### &nbsp; `Step 3` &nbsp; — &nbsp; Launch the Mission Dashboard

```bash
cd frontend
npm install
npm run dev
```

> 🌐 &nbsp; 3D Mission Dashboard live at **`http://localhost:5173`**

<br/>

---

<br/>

## 🔬 &nbsp; Tech Stack

<br/>

<div align="center">

| Layer | Technology | Role |
|:---|:---:|:---|
| 🪐 **Orbital Mechanics** | `python-sgp4` | TLE → precise 6D state vector propagation |
| 🔍 **Object Detection** | YOLOv8 | Sub-10cm debris detection with noise simulation |
| 📊 **State Tracking** | Kalman Filter / DeepSORT | Multi-object real-time state estimation |
| 🧠 **Trajectory Prediction** | PyTorch LSTM | N-step future orbital position forecasting |
| ⚡ **Collision Search** | SciPy KDTree | Sub-second proximity queries at scale |
| ⚖️ **Risk Classification** | XGBoost | Explainable conjunction risk tiering |
| 🤖 **Avoidance Planning** | Stable-Baselines3 PPO | Fuel-optimal ΔV maneuver computation |
| 🌐 **Backend API** | FastAPI + WebSockets | Async real-time AI inference engine |
| 🔀 **Proxy Layer** | Node.js | WebSocket multiplexing + state caching |
| 🖥️ **Visualization** | React 18 + Three.js | 3D globe · 60 FPS · 1,500+ objects |
| 📦 **Data Export** | JSON Schema | Conjunction event dataset logging |

</div>

<br/>

---

<br/>

<div align="center">

```
  ✦  ·  ·  ·  ·  ·  ·  ·  ✦  ·  ·  ·  ·  ·  ·  ·  ✦  ·  ·  ·  ·  ·  ·  ·  ✦
          Protecting the orbital commons — one conjunction at a time.
  ✦  ·  ·  ·  ·  ·  ·  ·  ✦  ·  ·  ·  ·  ·  ·  ·  ✦  ·  ·  ·  ·  ·  ·  ·  ✦
```

<br/>

[![NASA Space Apps](https://img.shields.io/badge/🚀_NASA-Space_Apps_Challenge-0B3D91?style=for-the-badge)](https://www.spaceappschallenge.org)

*Built with ❤️ for the orbital debris challenge · MIT Licensed*

</div>
