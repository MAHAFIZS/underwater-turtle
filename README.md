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

## 🔌 ROS2 Integration (Implemented)

This project now includes a ROS2 package for running the PN controller on a TurtleBot3
with LaserScan-based obstacle avoidance and stability protections.

### 📦 ROS2 Package Location


ros2/underwater_turtle_ros/


### ✅ What’s Included

- PN pursuit controller (LOS-based steering) reused from the project controller logic
- LaserScan front-cone obstacle detection (supports 0..2π scans)
- Smooth obstacle blending (no stop-go jitter)
- Side-hold hysteresis (prevents left-right flip-flop)
- Contact-freeze safety (prevents pushing into obstacles / flipping)
- Launch + params config (YAML)

### 🧪 Run (ROS2)

Build:

```bash
cd <repo_root>/ros2
colcon build --symlink-install
source install/setup.bash

Run node:

ros2 run underwater_turtle_ros pn_node

Or launch with parameters:

ros2 launch underwater_turtle_ros pn.launch.py
⚙️ Key Parameters (ROS2)

Parameters are in:

ros2/underwater_turtle_ros/config/pn_params.yaml

Important ones:

front_center_rad: scan front direction (usually 0.0 for TurtleBot3 0..2π scans)

obs_slow_dist, obs_stop_dist: avoidance blend range

contact_dist, contact_hold_s: safety stop on near-contact

v_max, omega_abs_max: stability caps

📌 Notes

TurtleBot3 LaserScan often reports angle_min=0, angle_max≈2π. In that case:

front is typically 0 rad, back is π rad.

The node is designed to be stable near obstacles (prevents oscillation and flipping).


### Also update this line near the top
Your README currently says “ROS2 integration – next step”. Change to:

- “ROS2 integration – implemented”

---

## After editing README, commit + push
Do this from **your real repo** (PowerShell in `D:\underwater-turtle` OR WSL in `/mnt/d/underwater-turtle`):

```bash
git add README.md
git commit -m "Update README with ROS2 integration"
git push

👤 Author

M. A. Hafiz
Robotics & Assistive Systems Research
Germany

GitHub: https://github.com/MAHAFIZS

📌 License

MIT License