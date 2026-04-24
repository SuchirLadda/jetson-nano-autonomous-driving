# Jetson Nano Autonomous Driving System

Real-time autonomous vehicle system built on an ARM-based NVIDIA Jetson Nano using ROS. The system integrates multi-sensor perception, state estimation, and control to enable safe navigation and obstacle avoidance.

---

## Overview

This project implements a full autonomy pipeline for a small-scale vehicle, combining LiDAR, camera, IMU, and odometry data to generate real-time driving decisions.

The system processes sensor data to estimate vehicle state and applies optimization-based control to safely navigate through environments with obstacles.

---

## Vehicle Platform

![Autonomous Vehicle](./images/car_front.jpg)

![Autonomous Vehicle](./images/car_side.jpg)

---

## System Architecture

The system is structured as a modular ROS pipeline:

- **Perception:** LiDAR, camera, and IMU data processing  
- **State Estimation:** Sensor fusion using odometry and onboard measurements  
- **Planning:** Gap detection and obstacle-aware navigation  
- **Control:** Real-time steering and speed control using optimization and feedback control  

---

## Key Features

- Real-time embedded system on NVIDIA Jetson Nano (Linux)
- Multi-sensor fusion (LiDAR, camera, IMU, odometry)
- ROS-based modular architecture
- Quadratic programming-based obstacle avoidance (virtual barrier method)
- Feedback-linearizing and PD control for vehicle motion
- Real-time debugging using ROS tools and logging

---

## Navigation & Control

### Virtual Barrier Navigation (QP-Based)

A quadratic program is solved in real time to generate safe steering commands by enforcing constraints derived from nearby obstacles.

- Identifies safe driving gaps using LiDAR data  
- Constructs virtual barriers around obstacles  
- Computes optimal steering direction subject to safety constraints  

---

### Control System

- Feedback-linearizing controller for trajectory tracking  
- PD controller for stability and responsiveness  
- Speed modulation based on obstacle proximity  

---

## Technologies Used

- C/C++, Python  
- ROS (Robot Operating System)  
- Linux (Jetson Nano)  
- LiDAR, IMU, camera integration  
- Real-time optimization (Quadratic Programming)  

---

## Development Context

Developed as part of McMaster University’s  
**3EY4 Electrical Systems Integration Project**

