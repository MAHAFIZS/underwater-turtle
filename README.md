# 🐢 Underwater Turtle – Pursuit Control Under Flow Fields

A modular simulation framework for evaluating pursuit control strategies
(Pure Pursuit vs Proportional Navigation) in dynamic current fields
(vortex, shear, gust).

Designed for:
- Controller research
- Robustness evaluation
- ROS2 integration
- Academic experimentation

---

## 🎯 Project Overview

This project simulates an underwater robot ("Turtle") attempting to capture
a maneuvering target ("Fish") under spatially varying flow fields.

Key features:

- 🐟 Stochastic fish motion (burst + wander behavior)
- 📡 Sonar measurement model with dropout + noise
- 📈 EKF-based target state estimation
- 🌊 Spatial current fields (vortex, shear, gust)
- 🎮 Modular controller interface
- 📊 Grid-based robustness evaluation
- 🧱 ROS-ready architecture

---

## 🧠 Controllers

### 1️⃣ Pure Pursuit
- Heading aligned to estimated target position
- Distance-based speed scaling
- Simple, stable baseline

### 2️⃣ Proportional Navigation (PN)
- LOS rate-based steering
- Closing-speed gated PN term
- Command slew rate limiting
- EMA-filtered LOS rate
- Tuned parameters (v1.0 freeze):


pn_N = 6.57
pn_k_los = 4.57
pn_vc_gate = 0.05
omega_cmd_rate_max = 2.0
pn_los_rate_alpha = 0.6


---

## 🏗 Architecture (ROS-Ready Freeze v1.0)


src/
├── controllers/
│ ├── pure.py
│ └── pn.py
├── sim/
│ └── episode.py
├── eval/
│ └── grid.py
├── turtle_robot.py
└── current_field.py

demos/
├── day5_heatmap.py
└── day6_compare.py


### Design Principle

- `controllers/` → control laws only
- `sim/` → physics + sensing + estimation
- `eval/` → benchmarking
- `demos/` → CLI wrappers only

This separation allows direct ROS integration
without modifying simulation logic.

---

## 📊 Evaluation Methodology

We perform grid-based robustness evaluation over:

- Current strength (flow magnitude)
- Current compensation level

For each grid cell:
- Run N stochastic episodes
- Measure capture rate
- Compute mean capture time

Example:


py -m demos.day5_heatmap --field vortex --controller pn --episodes 40


Outputs:


results/day5/
├── grid_metrics_<field><controller>.csv
└── heatmap_capture_rate<field>_<controller>.png


---

## 🧪 Capture Definition

Capture occurs when:

- Distance < r_capture
- Maintained for ≥ t_hold seconds

Default:

r_capture = 0.35 m
t_hold = 0.10 s


---

## 📈 Example Results (v1.0)

Observations:

- PN consistently outperforms Pure Pursuit in vortex fields
- PN improves robustness under high-current shear
- Closing-speed gating prevents PN over-steering
- Slew limiting prevents oscillatory snap behavior

---

## 🔒 Freeze Version

Current frozen release:


v1.0 – ROS-ready controller architecture


Tagged in Git as:

git tag v1.0


This version guarantees:

- Stable PN tuning
- Reproducible grid evaluation
- Controller state reset per episode
- Clean architecture separation

---

## 🚀 Running the Project

### 1️⃣ Setup


python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt


(Requires: numpy, matplotlib)

---

### 2️⃣ Run Evaluation


py -m demos.day5_heatmap --field vortex --controller pure
py -m demos.day5_heatmap --field vortex --controller pn


---

### 3️⃣ Compare Controllers


py -m demos.day6_compare --episodes 40 --fields vortex,shear


---

## 🔌 ROS Integration (Next Step)

Because controllers are isolated in:


src/controllers/


A ROS2 node can directly instantiate:

```python
controller = PNController()
omega_cmd, v_cmd = controller.compute(robot, dt, current)

No evaluation code is required in ROS.

📚 Research Context

This framework is useful for:

Marine robotics research

Interception under flow disturbances

Robust guidance law evaluation

Navigation under environmental drift

Comparative controller benchmarking

👤 Author

M. A. Hafiz
Robotics & Assistive Systems Research
Germany

GitHub: https://github.com/MAHAFIZS

📌 License

MIT License


---

# Next Step

After saving this:

```powershell
git add README.md
git commit -m "Add professional README"
git push