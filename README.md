# Optimizing Traffic Signals using Deep Q-Networks (DQN) in SUMO for Lagos Traffic

This repository contains the implementation of an intelligent traffic signal optimization system using Deep Q-Networks (DQN) within the SUMO (Simulation of Urban MObility) environment. The project aims to reduce traffic congestion and improve flow at intersections, specifically focusing on traffic patterns inspired by Lagos, Nigeria.

## Table of Contents

- [Aim](#aim)
- [Objectives](#objectives)
- [Introduction](#introduction)
- [Features](#features)
- [Agents Implemented](#agents-implemented)
- [Results and Analysis](#results-and-analysis)
- [Contributing](#contributing)

## Aim
To reduce traffic congestion, improve vehicle throughput, and minimize travel time and environmental impact in high-traffic areas of Lagos, specifically the Oke Afa Roundabout and Isolo–Ikotun Corridor, by implementing Reinforcement Learning (RL) as an alternative to conventional fixed-time signal systems.

## Objectives
-	To simulate real-world traffic conditions of the Oke Afa Roundabout and its extensions using the SUMO (Simulation of Urban MObility) environment.
-	To develop and implement a Deep Q-Learning (DQN) agent for adaptive traffic signal control.
-	To compare the performance of the intelligent (RL-controlled) system against a Fixed-Time traffic light baseline and a simpler Q-Learning (Q-table) approach across key metrics.


## Introduction

Urban traffic congestion is a pervasive problem, particularly in rapidly growing cities like Lagos. Traditional fixed-time traffic signal systems often fail to adapt to dynamic traffic conditions, leading to inefficiencies. This project explores the application of Deep Reinforcement Learning (DRL), specifically Deep Q-Networks, to create adaptive traffic signal controllers that can learn optimal signal timings in real-time.

SUMO is used as the traffic simulation platform, providing a realistic environment to train and evaluate the DQN agent.

## Features

-   **Deep Q-Network (DQN) Agent:** Implements a DQN model that learns optimal traffic light phases.
-   **Fixed-Time Agent:** A baseline comparison agent for evaluating performance.
-   **Q-Learning Agent:** Another reinforcement learning baseline for comparison.
-   **SUMO Simulation Integration:** Seamless interaction with SUMO for realistic traffic scenarios.
-   **Data-Driven Optimization:** The DQN agent learns from real-time traffic state (queue lengths, waiting times).
-   **Performance Metrics:** Tracks and visualizes key traffic metrics such as cumulative reward, queue length, and average waiting time.
-   **Reproducible Environment:** Project structure designed for easy setup and replication.

## Agents Implemented

### 1. Deep Q-Network (DQN) Agent

This agent uses a neural network to approximate the Q-values for different traffic light phases. It learns to make decisions that maximize cumulative reward, typically defined by minimizing vehicle waiting times and queue lengths.

### 2. Fixed-Time Agent (Baseline)

A simple agent that applies predefined, static traffic light timings. This serves as a fundamental benchmark to evaluate the performance of adaptive agents.

### 3. Q-Learning Agent (Baseline)

A basic reinforcement learning agent that uses a Q-table to store and update Q-values. While less scalable than DQN for large state spaces, it provides a valuable comparison for the learning process.

## Results and Analysis

After running the simulations, the `results/` directory will contain:

-   `Traffic metrics report.xlsx`: A spreadsheet detailing various traffic flow metrics (e.g., average waiting time, total travel time) for each simulation run.
-   Plots for each agent (e.g., `Fixed Time_Queue Lenght.png`, `Fixed Timing_reward image.png`): Visualizations of key performance indicators over the simulation duration. These plots help in comparing the effectiveness of the different agents.

## Contributing

Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.
