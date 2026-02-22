# AI-Powered Space Debris Tracking and Avoidance System (Research Grade)

## Problem Statement
To develop an AI-powered system for space debris tracking, trajectory prediction, collision risk assessment, and autonomous avoidance using physics-based simulation and simulated detection inputs.

## System Architecture
The pipeline enforces a strict 5-stage orchestration loop alongside the autonomous interactive simulation sandbox mapping physical components correctly:

```mermaid
graph TD
    classDef dataset fill:#2d3436,stroke:#74b9ff,stroke-width:2px,color:white;
    classDef physics fill:#2d3436,stroke:#00b894,stroke-width:2px,color:white;
    classDef ai fill:#2d3436,stroke:#a29bfe,stroke-width:2px,color:white;
    classDef viz fill:#2d3436,stroke:#fdcb6e,stroke-width:2px,color:white;

    subgraph Data Input Layer
        TLE[(CelesTrak TLE Data)]:::dataset
        CX[(Conjunction Data)]:::dataset
        PROXY[(Proxy Image Data)]:::dataset
    end

    subgraph Physics / State Engine
        PARSE[Dynamic TLE Parser]:::physics
        SGP4[SGP4 Orbital Propagator]:::physics
        ENV[3D State Environment]:::physics
        
        TLE --> PARSE
        PARSE --> SGP4
        SGP4 --> ENV
    end

    subgraph AI Risk Assessment Pipeline
        YOLO[YOLOv8 Detection]:::ai
        KALMAN[6D Kalman Filter]:::ai
        LSTM[Seq2Seq Trajectory LSTM]:::ai
        XGB[XGBoost Risk Assessor]:::ai
        PPO[PPO Evasive RL Agent]:::ai

        PROXY --> YOLO
        YOLO --> KALMAN
        KALMAN --> LSTM
        LSTM --> XGB
        CX --> XGB
        XGB -->|High Risk| PPO
    end

    subgraph Autonomous Removal Simulator
        TARGET[Proximity Target Selector]:::ai
        PLAN[Spline Trajectory Planner]:::ai
        
        ENV --> TARGET
        TARGET --> PLAN
        PLAN -->|Removal Velocity| ENV
    end

    subgraph Web UI Layer
        PLOTLY[Plotly 3D Graph Render]:::viz
        ST[Streamlit Orchestrator]:::viz
        
        ENV --> PLOTLY
        PLOTLY --> ST
        PPO --> ST
    end
```

## Scientific Limitations
* **Simulated Datasets**: Due to classification and orbital sensing limits, no real-time true optical dataset exists for LEO space debris. The YOLOv8 CV engine demonstrates logic and pipeline stability utilizing proxy datasets, effectively operating as a synthetic simulation.
* **RL Constrained Environment**: The RL PPO agent is trained within a bounded structural environment focusing purely on the conjunction event.
* **Simplified Physics**: The orbit propagation utilizes SGP4 and numerical RK4 integration with $J_2$ perturbation and basic drag models; higher-order spherical harmonics are abstracted for computation.

## Future Work
* Integration with live Space Situational Awareness (SSA) radar catalogs (e.g., Space Track).
* Multi-sensor fusion simulating both ground-radar latency and onboard optical tracking.
* High-fidelity orbital perturbation modeling (e.g., N-Body forces from Lunar/Solar gravity).
* Real-time embedded edge-hardware constraints analysis for satellite deployments.
