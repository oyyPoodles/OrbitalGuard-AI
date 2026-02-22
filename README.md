# AI-Powered Space Debris Tracking and Avoidance System (Research Grade)

## Problem Statement
To develop an AI-powered system for space debris tracking, trajectory prediction, collision risk assessment, and autonomous avoidance using physics-based simulation and simulated detection inputs.

## System Architecture

The pipeline strictly enforces a 5-stage orchestration loop, mimicking real-world collision avoidance operational logic:

1. **Simulated Detection (YOLOv8):** Represents an onboard optical sensing surrogate operating in a simulated environment. Real debris detection relies on phased-array radar and optical telescopes; because public space-grade optical datasets are unavailable, this module proves the computer vision integration logic via a proxy dataset.
2. **State Estimation (Kalman Filter):** Takes simulated 2D detection mappings and mathematically fuses them into a stable 6D state estimator ($X, Y, Z, V_x, V_y, V_z$), dropping Gaussian sensor noise.
3. **Trajectory Prediction (LSTM):** Evaluates the historical time-series positional vectors to accurately forecast the $t+1$ trajectory of the target object.
4. **Collision Risk Assessment (XGBoost):** Consumes relative spatial metrics (e.g., predicted miss distance, closure velocity) to classify the conjunction probability (HIGH, MEDIUM, LOW).
5. **Autonomous Avoidance (PPO RL):** A Deep Reinforcement Learning agent (Proximal Policy Optimization) engineered within a custom Gymnasium environment to calculate the precise $\Delta V$ escape maneuver.

## Scientific Limitations
* **Simulated Datasets**: Due to classification and orbital sensing limits, no real-time true optical dataset exists for LEO space debris. The YOLOv8 CV engine demonstrates logic and pipeline stability utilizing proxy datasets, effectively operating as a synthetic simulation.
* **RL Constrained Environment**: The RL PPO agent is trained within a bounded structural environment focusing purely on the conjunction event.
* **Simplified Physics**: The orbit propagation utilizes SGP4 and numerical RK4 integration with $J_2$ perturbation and basic drag models; higher-order spherical harmonics are abstracted for computation.

## Future Work
* Integration with live Space Situational Awareness (SSA) radar catalogs (e.g., Space Track).
* Multi-sensor fusion simulating both ground-radar latency and onboard optical tracking.
* High-fidelity orbital perturbation modeling (e.g., N-Body forces from Lunar/Solar gravity).
* Real-time embedded edge-hardware constraints analysis for satellite deployments.
