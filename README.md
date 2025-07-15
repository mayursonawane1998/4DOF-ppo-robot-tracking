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
**🧪 Primary Research Objectives**

- 🔧 **Implementation and Validation**:  
  To implement the PPO algorithm for controlling a 4-DOF robotic arm and validate its performance in a baseline scenario without visual feedback. This involves defining the **state and action spaces**, designing an appropriate **reward function**, and conducting **extensive training** to achieve reliable baseline performance.

- 👁️ **Impact Analysis of Visual Feedback**:  
  To evaluate the influence of **visual feedback from a camera mounted on the end effector** on the robot’s performance. This includes processing the visual data, integrating it with the simulation data into the PPO framework, and analyzing the robot’s ability to **find and follow the cube** using this combined input. The goal is to understand how visual perception affects robotic behavior and performance.

- 🧠 **Effectiveness of YOLO Integration**:  
  To investigate the effectiveness of incorporating **YOLO** for real-time cube detection and its impact on the robot’s tracking performance. This involves modifying the PPO algorithm to utilize the cube’s detected position, assessing the robot’s ability to follow the cube in **dynamic environments**, and comparing these results with the previous two setups.

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
   git clone https://github.com/mayursonawane1998/4DOF-ppo-robot-tracking.git
   cd 4DOF-ppo-robot-tracking

2. **Create a Python virtual environment (optional but recommended)**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt

## 🚀 How to Use:
1. **▶️ Train the PPO Agent**
Run the training loop:
   ```bash
   python ppo/train.py

2. **🎮 Test the Trained Agent**
Run the trained policy in simulation:
   ```bash
   python ppo/test.py

## 🔧 Tools & Libraries Used
Python 3.8+

PyBullet

Stable-Baselines3

NumPy

Matplotlib

Gym

SolidWorks (for URDF export) and More.

## 📚 Thesis & Documentation
The complete technical details, system architecture, algorithm design, and results are available in the full Master’s thesis report.

## Thank you for checking out my thesis project!













