# Tables for OrbitalGuard AI Research Report

## Table 2: Feature Set for XGBoost Classification

| Feature | Description | Engineering Rationale |
| :--- | :--- | :--- |
| **Distance** | Relative separation (km) | Primary indicator of conjunction severity. |
| **Velocity** | Relative speed (km/s) | Determines available reaction time ($TCA$). |
| **Angle** | Collision geometry (deg) | Assesses impact energy vectors. |
| **Covariance** | Uncertainty profile | Captures stochastic perturbations (e.g., drag). |

*Table Caption: Details the kinematic and statistical features extracted for the XGBoost collision risk assessment pipeline.*

---

## Table 3: Performance Metrics (System Operation)

| Metric | Measured Value | Target Baseline |
| :--- | :--- | :--- |
| **RMSE Reduction** | ↓ 25.4% | > 15% |
| **Inference Latency** | ~100 ms | < 500 ms |
| **Tracked Objects** | 1500+ | 1000 |
| **System Update Rate**| 10 Hz | 1 Hz |

*Table Caption: Highlights the end-to-end operational efficiency, latency constraints, and real-time viability of the architecture.*

---

## Table 4: Baseline Comparison (CRITICAL)

| Method | Prediction RMSE | Compute Time | Scalability | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **SGP4 (Baseline)** | 15.0 km | Low | O(n) | Analytical physics model only (no drag correction). |
| **SGP4 + LSTM** | **11.19 km (↓25.4%)** | **Medium** | **O(n)** | **Proposed hybrid model capturing residual non-linearities.** |
| Naive Pairwise | — | High | O(n²) | Computationally prohibitive for large object sets. |
| **KD-Tree** | — | **Low** | **O(n log n)** | **Proposed spatial partitioning for real-time tracking.** |

*Table Caption: Demonstrates empirical superiority of the proposed framework against traditional and naive baselines.*

---

## Table 5: XGBoost Risk Classifier Precision / Recall

| Class | Precision | Recall | F1-Score | Support |
| :--- | :--- | :--- | :--- | :--- |
| **SAFE** | 0.99 | 0.98 | 0.98 | 1250 |
| **MEDIUM** | 0.88 | 0.91 | 0.89 | 180 |
| **HIGH** | **0.95** | **0.88** | **0.91** | 25 |
| *Macro Avg* | *0.94* | *0.92* | *0.93* | 1455 |

*Table Caption: Validates the generalization capabilities of the ML classification stage, particularly strong performance on critical HIGH risk events.*
