# 4DOF-ppo-robot-tracking
Master's thesis project: PPO-based control of 4DOF robot in PyBullet to track a target cube.
# 🤖 4DOF PPO Cube Tracking — Master's Thesis

This project presents a reinforcement learning approach using **Proximal Policy Optimization (PPO)** to control a **custom 4-DOF  robotic arm** for **real-time cube tracking** in the **PyBullet** simulation environment. The robot model was exported from **SolidWorks** and integrated into a learning framework for robotic control using PPO.

> 🎓 Master's Thesis – Vrije Universiteit Brussel (VUB) & Université libre de Bruxelles (ULB)  
> 👨‍🔬 Author: Mayur Ashok Sonawane  
> 🗓️ Academic Year: 2023–2024  

---

## 🎯 Project Objectives

- Develop a simulation environment for a 4-DOF robot using URDF in PyBullet.
- Use Proximal Policy Optimization (PPO) for continuous control of the robot.
- Train the robot to track the position of a moving cube.
- Evaluate training performance using reward curves and visual results.
- Create a modular base for future work in robotics, control, and sim-to-real transfer.

---

## 🧠 Technical Overview

- **Environment**: PyBullet  
- **Robot Model**: custom 4-DOF version (URDF from SolidWorks)  
- **Algorithm**: PPO (from Stable-Baselines3)  
- **Tracking Goal**: Robot end-effector follows a cube's position  
- **Observation**: Joint positions, velocities, and target cube coordinates  
- **Action Space**: Continuous joint velocity control  
- **Reward**: Distance-based negative reward for tracking accuracy

---

## 📸 Demo (Coming Soon)

Videos, GIFs, and screenshots will be uploaded to show robot performance after training.

---

## 📁 Project Directory Structure

4DOF-ppo-robot-tracking/
│
├── urdf/ # Custom 4DOF robot URDF files
├── assets/ # Cube model or mesh files
├── ppo/ # PPO training and evaluation scripts
│ ├── train.py # Training loop
│ ├── test.py # Evaluate trained agent
│ └── ppo_agent.py # PPO configuration
│
├── simulation/ # Environment wrapper and robot-cube interaction
├── results/ # Trained models, logs, reward plots
├── requirements.txt # Python dependencies
├── README.md # Project overview (this file)
---

## ⚙️ Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/ur5e-ppo-robot-tracking.git
   cd ur5e-ppo-robot-tracking

2. **Create a Python virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

## 🚀 How to Use:

1. **▶️ Train the PPO Agent**---
Run the training loop:
   ```bash
   python ppo/train.py

2. **🎮 Test the Trained Agent**---
Run the trained policy in simulation:
   ```bash
   python ppo/test.py










