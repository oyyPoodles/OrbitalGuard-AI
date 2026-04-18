# OrbitalGuard AI - System Architecture

Here is the visual pipeline flow diagram requested for your presentation. You can screenshot this or embed the Mermaid code directly into your slide deck tooling.

```mermaid
graph TD
    classDef data fill:#2C3E50,stroke:#34495E,stroke-width:2px,color:#ECF0F1;
    classDef physics fill:#E74C3C,stroke:#C0392B,stroke-width:2px,color:#FFF;
    classDef ai fill:#8E44AD,stroke:#732D91,stroke-width:2px,color:#FFF;
    classDef filter fill:#F39C12,stroke:#D68910,stroke-width:2px,color:#FFF;
    classDef ui fill:#2980B9,stroke:#2471A3,stroke-width:2px,color:#FFF;

    A[📡 Live TLE Ephemeris Data<br/>CelesTrak API]:::data --> B(🧮 SGP4 Orbital Propagator <br/>Deterministic Physics Base):::physics
    B --> C(🧠 LSTM Neural Network <br/>Temporal Residual Drift Correction):::ai
    C --> D{🔍 KDTree Classifier <br/>Spatial Search & Filtering}:::filter
    D -->|O N log N Filtering| E(🛡️ XGBoost Classifier <br/>Risk Factor: LOW / MED / HIGH):::ai
    E --> F[🤖 PPO RL Agent <br/>Autonomous ΔV Thrust Planning]:::ai
    F --> G[💻 React + WebGL Client <br/>Live 3D Dashboard Data]:::ui
```

### Presentation Talking Points
- **Blue (Data):** System automatically fetches and caches live updates.
- **Red (Physics):** The `SGP4` algorithm computes the mathematical standard trajectory.
- **Purple (AI/ML):** `LSTM` corrects the mathematical error. `XGBoost` categorizes the risk. `PPO` executes collision avoidance.
- **Orange (Filter):** The `KDTree` ensures the entire pipeline happens in real-time by discarding irrelevant distances to save compute power.
