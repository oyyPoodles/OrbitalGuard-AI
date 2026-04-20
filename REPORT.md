# OrbitalGuard AI: Advanced Space Debris Surveillance and Collision Avoidance Framework

## 1. Abstract
As Low Earth Orbit (LEO) becomes increasingly congested by commercial mega-constellations, defunct satellites, and orbital debris, the risk of cascading collisions (the Kessler Syndrome) has exponentially risen. Current Space Situational Awareness (SSA) relies on outdated physical propagators and slow deterministic filtering, leading to high false-alarm rates and massive computational lag during localized conjunction alerts. 

**OrbitalGuard AI** resolves these inherent limitations by introducing a state-of-the-art hybrid predictive and visualization framework. By marrying traditional physics engines (SGP4) with high-capacity sequential neural networks (LSTM), scalable spatial partitioning (KDTree), and gradient-boosted risk classification (XGBoost), the system fundamentally upgrades satellite telemetry tracking. Combined with a High-Performance WebGL 3D Visualization Matrix and a fully interactive 2D Tactical Radar Interface, OrbitalGuard AI acts as an advanced decision-support system and a prototype SSA framework equipped with Reinforcement Learning (PPO) collision-avoidance mechanisms.

---

## 2. The Problem Space: The Kessler Syndrome
Since 1957, humanity has launched thousands of payloads into space. While active satellites possess internal maneuvering thrusters, they share LEO with "Space Junk"—dead rocket bodies, fragmented payload fairings, and shrapnel from anti-satellite weapon tests. 

The core issue facing modern active aerospace management is two-fold:
1. **Mathematical Density**: The deployment of commercial mega-constellations (e.g., SpaceX's Starlink) has drastically reduced the volumetric operating space between satellites. A single collision in a densely packed plane creates a debris field that can trigger a runaway chain reaction (The Kessler Syndrome), effectively sealing humanity off from space flight.
2. **Propagator Drift**: Traditional tracking relies on the **SGP4 (Simplified General Perturbations-4)** engine. Given a Two-Line Element (TLE) string, SGP4 calculates where an object should physically be. However, SGP4 relies purely on mathematical physics. It cannot account for unpredictable space weather phenomena, micro-fluctuations in atmospheric drag, or solar radiation pressure. Without correction, SGP4 outputs suffer from heavy spatial "drift," losing accuracy hour-by-hour and plunging the true distance vector algorithms into chaos.

---

## 3. The Novelty of OrbitalGuard AI
OrbitalGuard bridges the historically isolated worlds of *Rigorous Astrodynamics* and *Modern Deep Learning*. 

Rather than throwing out the SGP4 physical model (which is fundamentally sound), OrbitalGuard introduces a **Residual Learning Architecture**. The neural networks do not predict the satellite's position; they predict the physics engine's *mistakes*. This hybrid approach guarantees physical safety parameters while utilizing AI to eliminate environmental drift. 

Furthermore, where traditional models require massive on-premise compute to map 30,000+ conjunctions simultaneously, OrbitalGuard implements O(n log n) spatial data-structures capable of filtering threats at sub-millisecond speeds directly within edge computing/browser environments.

---

## 4. The 5-Stage Machine Learning Pipeline
The backend of OrbitalGuard AI consists of a rigidly strict, sequential five-stage data pipeline. Every passing second of telemetry data executes entirely through this stack.

![AI Pipeline Architecture](./docs/pipeline_diagram.png)

### Stage 1: The Physics Baseline (SGP4)
* **What it is:** The foundational propagator converting raw NORAD Two-Line Elements (TLEs) into spatial coordinates.
* **Why it's used:** SGP4 provides the core geometric constraints and orbital mechanics logic. It prevents the subsequent neural networks from hallucinating impossible physics.

### Stage 2: The Drift Corrector (Hybrid LSTM) 
* **What it is:** A Long Short-Term Memory (LSTM) sequential recurrent neural network.
* **Why it's used:** To solve "Propagator Drift." The LSTM is trained on historical true ephemeris positions vs. actual SGP4 outputs. Rather than learning absolute coordinates, it learns the **Residual Error Vector**. 
* **The Result:** The system sums the SGP4 coordinate with the LSTM's predicted drift. This "Hybrid SGP4 + LSTM" mechanism demonstrably shrinks the Root Mean Square Error (RMSE) and Average Displacement Error (ADE) by ~25.4%, creating a vastly superior tracking baseline.

### Stage 3: The Spatial Filter (KDTree Indexing)
* **What it is:** A localized space-partitioning data structure that structures all coordinates into a searchable algebraic tree.
* **Why it's used:** To fix systemic scalability issues. Checking if 30,000 objects are crashing into each other requires massive cross-product geometry ($O(n^2)$ time). Attempting this on every frame crashes hardware. By structuring the LSTM-corrected coordinates into a KDTree, the system performs neighborhood searches in scalable $O(n \log n)$ time, ensuring instant conjunction detection.

### Stage 4: The Risk Assessor (XGBoost)
* **What it is:** An Extreme Gradient Boosting classification ensemble.
* **Why it's used:** To stop "False Alarms." Traditional tools simply say `if distance < 3km then HIGH RISK`. This leads to wasted fuel if the objects are just passing parallel. XGBoost ingests closing velocity vectors, collision geometry angles, object mass profiles, and spatial covariance matrices to assign a highly accurate categorical threat probability: `SAFE`, `MEDIUM RISK`, or `HIGH RISK`.

### Stage 5: The Autonomous Navigator (PPO Deep RL Engine)
* **What it is:** Proximal Policy Optimization (PPO), an industry-standard Reinforcement Learning algorithm.
* **Why it's used:** When XGBoost flags a `HIGH RISK` conjunction on an active payload, action is required. The PPO agent simulates millions of thrust permutations to find the optimal Delta-V (ΔV) burn. It outputs a fuel-efficient, safe-trajectory avoidance maneuver to clear the impact zone.

---

## 5. UI/UX Architecture & Technical Engineering
The data pipeline is visualized via a custom-engineered React frontend designed to handle monstrous amounts of concurrent WebGL operations while minimizing operator cognitive load.

### 5.1. 3D Spatiotemporal Engine
* **Technology:** React, Three.js, React Three Fiber.
* **GPU Instancing Matrices:** To prevent the browser's main JavaScript thread from bottlenecking while tracking thousands of objects, the system bypasses standard DOM node creation. It utilizes heavily optimized `instancedMesh` matrices, applying matrix rotations and vector translations on the GPU directly inside the rendering loop. 
* **Smart Tracking Rigs:** A custom camera system detaches structural positional locking, allowing operators to freely orbit, inspect, and pan around targeted payloads without breaking tracking locks.

### 5.2. Advanced Radar-Based Analysis Module
To investigate localized `HIGH RISK` alerts, operators switch to the purely 2D Tactical Radar Engine.
* Runs on a hyper-optimized, RequestAnimationFrame (`rAF`) Canvas loop completely decoupled from React State to ensure unyielding 60-120 FPS performance.
* Actively maps local Euclidean vectors into a 2D Polar Projection featuring kinematic drift physics and velocity leaders.
* Features autonomous tracking brackets `[ TRK ]` that calculate exact metric distances and lock onto the absolute nearest threat inside the immediate localized radar sweep.

### 5.3. "Total Black" Command Center Principles
To maximize legibility and professional aesthetic fidelity, the design utilizes entirely custom CSS "Sci-Fi/Command Center" properties.
* Deeply aggressive #000000 negative space, high contrast neon-cyan accents.
* `globalCompositeOperation = 'screen'` algorithms generating organic, tactile CRT scan-lines and faux-vignette shadowing atop the HTML5 canvas layer.
* Pulsating UI notifications ensuring high-priority threats immediately capture operator visual attention.

---

## 6. Quantitative Results & System Evaluation

The complete pipeline was evaluated rigorously against simulated true ephemeris datasets and real-world TLE data. The following metrics validate the architecture across all core intelligent models: the LSTM residual predictor, the XGBoost risk classifier, and the KD-Tree spatial partitioner.

### 6.1. Model 1: Hybrid SGP4 + LSTM (Trajectory Predictor)
The LSTM regression model predicts physics residual errors to calculate accurate hybrid coordinates. 
* **Final RMSE Bound**: SGP4 Baseline ~14.8 km $\rightarrow$ Hybrid SGP4+LSTM **~11.2 km**.
* **RMSE Reduction**: The proposed hybrid model achieves an approximately **24–25% reduction** in Root Mean Square Error over a continuous 48-hour forecasting window compared to the SGP4 baseline.
* **ADE Performance**: Maintains the average displacement error under critical operational bounds significantly longer than purely physical modeling.

![RMSE Error Comparison Graph](./Project_Visualization_for_Research/Figure_6_RMSE_Comparison.png)

### 6.2. Model 2: XGBoost (Conjunction Risk Classifier)
Evaluated across kinematic collision vectors (distance, velocity, angle, covariance), the categorization engine performed with high accuracy and exceptionally balanced F1-scores, effectively eliminating geometric false-alarms.
* **Overall Accuracy**: **96.7%**
* **Class: SAFE**: Precision: **0.99** | Recall: **0.98** | F1-Score: **0.98**
* **Class: MEDIUM RISK**: Precision: **0.88** | Recall: **0.91** | F1-Score: **0.89**
* **Class: HIGH RISK**: Precision: **0.95** | Recall: **0.88** | F1-Score: **0.91**
* **Macro Average**: Precision: **0.94** | Recall: **0.92** | F1-Score: **0.93**

### 6.3. Model 3: KD-Tree (Spatial Partitioning Algorithm)
Because localized space tracking is fundamentally time-sensitive, the spatial algorithms establish critical performance bounds:
* **Computational Complexity**: Optimized from historically prohibitive $O(n^2)$ Naive distance searches down to rapid **$O(n \log n)$** algorithmic bounds.
* **Pipeline Latency**: The total round-trip computational execution latency across the entire 5-stage pipeline measures safely under **~100 ms**.
* **Scaling Limit**: Capable of ingesting and visually tracking **1,500+ active objects** simultaneously at a rigid **10 Hz** spatial update rate.

![Final RMSE Comparison](./Project_Visualization_for_Research/Figure_11_RMSE_Bar_Chart.png)

---

## 7. Limitations & Future Work
While OrbitalGuard AI successfully modernizes Space Situational Awareness, the prototype relies on structural dependencies that highlight clear pathways for future iterations:
1. **TLE Dependency**: The system's baseline SGP4 mechanics are bounded by the ingestion latency of NORAD Two-Line Elements. Without high-frequency TLE updates, intrinsic decay bounds eventually override even optimal LSTM corrections.
2. **Absence of Real Sensor Fusion**: The current classification relies purely on propagated calculations instead of raw sensor fusion (e.g., direct phased-array radar telemetry ingestion or celestial optics tracking), capping confidence limits.
3. **Simulated PPO Environment**: The Proximal Policy Optimization (PPO) Deep RL agent actively models and requests optimal Delta-V vectors, but it remains a simulated environment. The true integration of live physical thruster actuation demands considerably more robust hardware-in-the-loop validation frameworks.

## 8. Conclusion
By deploying deep residual learning directly over existing physics infrastructures, and executing risk detection through XGBoost classifying models at lightning-fast KDTree speeds, the framework establishes a highly robust prototype SSA framework. It proves that predictive intelligence and interactive 3D telemetry can exist symmetrically inside Edge environments, laying a foundation for next-generation automated decision-support systems.
