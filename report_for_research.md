# OrbitalGuard AI: Comprehensive Research Documentation

## 1. INTRODUCTION

### 1.1 Introduction to Project
OrbitalGuard AI is a prototype Space Situational Awareness (SSA) and decision-support framework designed to securely monitor, analyze, and predict collision events within Low Earth Orbit (LEO). By marrying rigorous astrodynamics (SGP4 propagation) with modern deep learning components (LSTM and XGBoost), the system provides an autonomous, real-time command dashboard capable of mapping hyper-velocity trajectories, identifying conjunctions, and simulating fuel-efficient avoidance maneuvers.

### 1.2 Problem Statement and Description
As commercial mega-constellations expand, LEO approaches critical spatial density. Traditional tracking models heavily rely on physics-based propagators, which suffer from systemic "drift" due to unpredictable space weather, leading to unacceptable margins of spatial error. The problem is simple but catastrophic: A single hyper-velocity collision generates an exponential debris field (The Kessler Syndrome), which could theoretically sever all global space operations. Current models lack the hybridized intelligence required to predict the drift residuals and autonomously compute avoidance geometries at sub-second speeds.

### 1.3 Motivation
The sheer density of sub-10cm "Space Junk" has rendered purely manual and strictly geometrical tracking models obsolete. The motivation behind OrbitalGuard AI is to bridge the gap between deterministic aerospace physics and predictive artificial intelligence, demonstrating that a real-time, browser-based edge computing environment can effectively serve as an aerospace defense matrix without requiring heavy macroscopic server farms.

### 1.4 Sustainable Development Goal of the Project
This project aligns directly with the United Nations **Sustainable Development Goal 9 (Industry, Innovation, and Infrastructure)**. Global telecommunications, banking, and GPS logistics rely completely on LEO satellite infrastructure. Ensuring the longevity and sustainable survival of our orbital environments is critical to preserving modern terrestrial society.

---

## 2. LITERATURE SURVEY

### 2.1 Overview of the Research Area
Research in Space Situational Awareness fundamentally relies on orbital mechanics and collision probability algorithms. Classical methodologies focus strictly on physical propagation from Two-Line Elements (TLEs), utilizing Cartesian transformations to predict bounding box overlaps. 

### 2.2 Existing Models and Frameworks
Existing frameworks, heavily utilized by the U.S. Space Force and NASA (e.g., SOCRATES), predominantly deploy standard Simplified General Perturbations (SGP4). Once coordinates are propagated, objects are fed through massive filtering nodes simulating deterministic distance tolerances (e.g., triggering alerts if objects close within a 5 km radius). 

### 2.3 Limitations Identified from Literature Survey (Research Gaps)
1. **Uncorrected Propagation Drift**: Classical algorithms fail to account for chaotic micro-density anomalies in the thermosphere, causing prediction vectors to "drift" continuously over time.
2. **Computational Scaling Problems**: Calculating conjunction distance vectors for 30,000 active objects relies on traditional cross-product matrices bounding in $O(n^2)$ time, which proves computationally crippling for real-time edge processing.
3. **Rigid Alert Thresholds**: Deterministic alerts result in excessive "False Positives," forcing satellites to expend critical maneuvering fuel avoiding objects that would safely pass by organically. 

### 2.4 Research Objectives
- To minimize SGP4 spatial trajectory error using LSTM Neural Networks.
- To execute conjunction proximity searches in scalable $O(n \log n)$ time bounds.
- To triage risk probabilities using dynamic Machine Learning classifications instead of rigid distance bounds.
- To rapidly visualize dense volumetric target tracking utilizing real-time GPU Instancing.

### 2.5 Product Backlog (Key user stories with Desired outcomes)
1. **As an Aerospace Engineer**, I need to visualize thousands of TLEs in standard 3D space, so that I can rapidly assess active regional orbits.
2. **As an AI Researcher**, I need an LSTM network capable of recognizing drift vectors, so that trajectory variables update continuously.
3. **As a Control Operator**, I need instant threat classification (SAFE, MEDIUM, HIGH), so I don't waste cognitive energy on non-threats.
4. **As a Mission Specialist**, I want the system to calculate collision avoidance paths via Reinforcement Learning, so I have a reliable Delta-V template.

### 2.6 Plan of Action (Project Road Map)
- **Phase 1**: Environment configuration, physics pipeline building, and raw TLE ingestion.
- **Phase 2**: AI integration (Training LSTM predictors, integrating KDTree indexing, and running XGBoost classification).
- **Phase 3**: Frontend architecture using React/Three.js focusing on GPU instancing.
- **Phase 4**: Tactical 2D Radar development and UX/UI aesthetic polishing.
- **Phase 5**: Real-world quantitative analysis and documentation.

---

## 3. METHODOLOGY

### 3.1 Physics-Based Orbit Propagation (SGP4)

The orbital trajectory is propagated using the SGP4 model based on TLE data of the ISS. As shown in Fig.~\ref{fig:sgp4}, the model produces state vectors in the Earth-Centered Inertial (ECI) frame over a 48-hour horizon.

\caption{Orbit propagation using the SGP4 model over a 48-hour horizon in the Earth-Centered Inertial (ECI) frame. The trajectory is generated from publicly available TLE data of the International Space Station (ISS), showing multiple orbital revolutions along with initial ($t_0$) and final ($t_{48}$) state vectors.}

### 3.2 Algorithm: Hybrid Orbital Collision Prediction and Avoidance Pipeline

The overall workflow of the proposed system is summarized in Algorithm 1.

\vspace{6pt}
\noindent\rule{\linewidth}{0.5pt}

\noindent\textbf{Algorithm 1: Hybrid Orbital Collision Prediction and Avoidance Pipeline}

\vspace{4pt}
\noindent\textbf{Input:} TLE data for $N$ space objects \\
\textbf{Output:} Collision risk labels and avoidance maneuvers

\vspace{4pt}
\noindent\textbf{Procedure:}

\begin{enumerate}
\item Propagate orbits using SGP4 to obtain $x_i^{SGP4}$
\item Estimate residual $\Delta x_i$ using LSTM
\item Compute corrected states $x_i^{hybrid} = x_i^{SGP4} + \Delta x_i$
\item Build KDTree using $\{x_i^{hybrid}\}$

\item For each object $i$, query neighbors within threshold $d_{th}$ and form candidate pairs $(i,j)$

\item For each candidate pair $(i,j)$:
\begin{itemize}
\item Extract features: distance, velocity, angle, covariance
\item Predict risk using XGBoost $\rightarrow$ \{SAFE, MEDIUM, HIGH\}
\end{itemize}

\item For each HIGH-risk pair $(i,j)$:
\begin{itemize}
\item Compute optimal maneuver $\Delta V$ using PPO
\end{itemize}

\item Return risk labels and maneuver actions
\end{enumerate}

\noindent\rule{\linewidth}{0.5pt}

---

## 4. SPRINT PLANNING AND EXECUTION METHODOLOGY

### 4.1 SPRINT I: Predictive Pipeline & Data Architecture

#### 3.1.1 Objectives with user stories of Sprint I
- Implement reliable mathematical SGP4 parsing using standard Python protocols.
- Build the Hybrid Neural Array for residual displacement correction.
- *User Story*: "As an automation backend, I need to parse raw NORAD structural elements instantly to prepare the data for visualization tracking."

#### 3.1.2 Functional Document
Sprint I primarily focuses on the algorithmic backend script structures (`SGP4 → LSTM → KDTree → XGBoost → PPO`). The core functionality verifies that telemetry sets ingest accurately, calculate basic Earth-Centered coordinates, and output into JSON API protocols efficiently.

#### 3.1.3 Architecture Document
![AI Pipeline Architecture](./docs/pipeline_diagram.png)  
*(The foundational mapping sequence implemented during Sprint I establishing the logic loop).*

#### 3.1.4 Outcome of objectives/ Result Analysis
The initial pass successfully calculated physical variables across 1,500 active payloads in ~100ms. The KDTree structural indexing proved inherently stable at $O(n \log n)$ searches, completely resolving the $O(n^2)$ bottleneck found in earlier testing.

#### 3.1.5 Sprint Retrospective
While physical calculations performed extremely well, early tests exposed rapid metric drift. This led to the fundamental shift toward adopting the LSTM "Residual Learning" approach to natively patch spatial loss over extended forecasting windows.

---

### 3.2 SPRINT II: Dashboard Visualization & Tactical UI Overhaul

#### 3.2.1 Objectives with user stories of Sprint II
- Inject the AI pipeline into a live WebGL 3D Visualization engine.
- Finalize the Advanced Radar "Command Center" aesthetic protocol.
- *User Story*: "As a mission specialist, I want immediate glowing visual indicators dictating XGBoost threat responses so I can react immediately."

#### 3.2.2 Functional Document
Sprint II transitioned directly to Frontend Client structuring (`App.jsx`, `AnalyticsDashboard.jsx`, etc.). It dictated the absolute necessity for decoupled graphic pipelines, isolating the React State from explicit HTML5 Canvas `.draw()` loops to guarantee 60-120 FPS parameters.

#### 3.2.3 Architecture Document
The visual hierarchy separates into two primary clusters: The `LiveFeed` 3D Mesh Engine (utilizing `@react-three/fiber` Instancing matrices) and the `AnalyticsDashboard` 2D Radar Overlay (handling military-grade HUD patterns, velocity lines, and threat bracket tracking).

#### 3.2.4 Outcome of objectives/ Result Analysis
Successfully blended organic electronic UI styling (Faux CRT overlays, neon cyan components) with rigorously tracked analytical matrices. The dashboard maintains highly stable performance regardless of the volume of data nodes actively loaded into the scene.

#### 3.2.5 Sprint Retrospective
A defining success. The decision to reject simple text-based data displays in favor of a full "Total Black Command Center" greatly streamlined operator focus, proving that dense AI statistics can be rendered elegantly.

---

## 6. RESULTS AND DISCUSSIONS

### 6.1 Project Outcomes (Performance Evaluation, Comparisons, Testing Results)

![RMSE Error Comparison Graph: SGP4 vs Hybrid LSTM](./Project_Visualization_for_Research/Figure_6_RMSE_Comparison.png)
\caption{RMSE comparison between SGP4 and the proposed hybrid SGP4+LSTM model over the prediction horizon, demonstrating consistent error reduction.}

![Final Prediction RMSE Comparison: Bar Chart](./Project_Visualization_for_Research/Figure_11_RMSE_Bar_Chart.png)
\caption{Final RMSE comparison showing improved accuracy of the hybrid model over the SGP4 baseline.}

1. **Tracking Precision Optimization (RMSE)**: Hybridizing the traditional SGP4 physical model with LSTM sequential algorithms demonstrably compressed target predictive drift. Across continuous 48-hour forecasting windows, the proposed hybrid model achieves approximately 24–25\% reduction in RMSE compared to the SGP4 baseline.
2. **Computational Real-Time Output**: 
   - Processing limit tests verified stable organization of **1,500+ dynamic LEO objects**.
   - Pipeline round-trip latency (from physics parse to Reinforcement execution analysis) stabilized at **~100 ms**.
   - Application execution refresh rates locked in at a minimum **10 Hz**.
3. **Classification Integrity**: Using the XGBoost classifier eliminated false geometric positives by ensuring categorical risk assignments actively evaluated constraint radii alongside exact kinetic velocity behaviors.

---

## 7. CONCLUSION AND FUTURE ENHANCEMENT

OrbitalGuard AI validates that utilizing Artificial Intelligence not to replace physics, but to actively correct residual error drift, produces vastly superior Space Situational Awareness frameworks. By scaling spatial logic at KDTree speeds and structuring data through an engaging, high-performance WebGL platform, defensive analysis can natively happen directly in secure Edge/Browser formats. 

**Future Enhancements:**
1. **Live Sensor Fusion Tracking**: Moving beyond propagation equations by actively pulling arrays from ground-based phased RF Radars into the model constraints.
2. **Direct Hardware Deployment**: Actively transitioning the Reinforcement PPO module from theoretical simulation loops into direct payload Delta-V actuation commands.
3. **Constellation Decentralization**: Applying the model actively onboard satellite computing matrices, allowing them to autonomously verify trajectories without Ground Control links.
