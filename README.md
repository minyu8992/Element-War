# Element-War: Pose-Driven Elemental Battle Game

## Overview
**Element War** is a real-time, pose-controlled two-player battle game that combines **Pose Estimation** and **Diffusion-Based Visual Generation**.  
Players perform specific poses to summon elemental attacks — *Water, Fire, Wind, Thunder, or Earth* — each with unique interactions following a rock-paper-scissors-like rule system.  
The game detects player poses using **NVIDIA trt_pose** and dynamically generates corresponding elemental visuals using a **Tiny Diffusion Model** in real time.

---

## Features
- **Pose-Driven Gameplay:** Real-time human pose detection triggers elemental attacks.  
- **Diffusion-Based Visuals:** Tiny Diffusion Model denoises and generates element visuals dynamically.  
- **Element Interaction Rules:**  
  - Water > Fire  
  - Fire > Wind  
  - Wind > Thunder  
  - Thunder > Earth  
  - Earth > Water  
- **Dual-Player Support:** Supports 1v1 gameplay with attack animations and health tracking.  
- **Edge AI Optimization:** Runs efficiently on **NVIDIA Jetson Xavier** with low latency.

---

## 🧩 Project Architecture


---

## Models

### Pose Estimation — `NVIDIA trt_pose`
- Based on **ResNet18 backbone** for real-time keypoint detection.  
- Extracts human joint positions and identifies predefined pose templates.  
- Used to trigger corresponding elemental attacks.

### Element Generation — `Tiny Diffusion Model`
- Lightweight diffusion model performing **reverse denoising** to generate element visuals.  
- Produces stylized representations of elemental attacks.  
- Optimized for fast inference to support real-time gameplay.

---

## Deployment on NVIDIA Jetson Xavier
- **Platform:** NVIDIA Jetson Xavier (Edge AI device).  
- **Optimizations:**
  - Converted models using **TensorRT** for accelerated inference.  
  - Used **half-precision (FP16)** for diffusion generation to reduce latency.  
  - Optimized frame processing pipeline with **OpenCV** for smooth gameplay.
- **Performance:** Achieved low-latency, real-time interaction between player gestures and visual effects.

---

## Demo

