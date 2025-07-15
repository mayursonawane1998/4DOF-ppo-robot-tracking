# 4DOF-ppo-robot-tracking
Master's thesis project: PPO-based control of 4DOF robot in PyBullet to track a target cube.
# 🤖 4DOF PPO Cube Tracking — Master's Thesis

This project presents a reinforcement learning approach using **Proximal Policy Optimization (PPO)** to control a **custom 4-DOF UR5e robotic arm** for **real-time cube tracking** in the **PyBullet** simulation environment. The robot model was exported from **SolidWorks** and integrated into a learning framework for robotic control using PPO.

> 🎓 Master's Thesis – Vrije Universiteit Brussel (VUB) & Université libre de Bruxelles (ULB)  
> 👨‍🔬 Author: Mayur Ashok Sonawane  
> 🗓️ Academic Year: 2023–2024  

---

## 🎯 Project Objectives

- Develop a simulation environment for a 4-DOF UR5e robot using URDF in PyBullet.
- Use Proximal Policy Optimization (PPO) for continuous control of the robot.
- Train the robot to track the position of a moving cube.
- Evaluate training performance using reward curves and visual results.
- Create a modular base for future work in robotics, control, and sim-to-real transfer.

---

## 🧠 Technical Overview

- **Environment**: PyBullet  
- **Robot Model**: UR5e – custom 4-DOF version (URDF from SolidWorks)  
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

