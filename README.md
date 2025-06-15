# Optimizing Traffic Signals using Deep Q-Networks (DQN) in SUMO for Lagos Traffic

This repository contains the implementation of an intelligent traffic signal optimization system using Deep Q-Networks (DQN) within the SUMO (Simulation of Urban MObility) environment. The project aims to reduce traffic congestion and improve flow at intersections, specifically focusing on traffic patterns inspired by Lagos, Nigeria.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Agents Implemented](#agents-implemented)
- [Results and Analysis](#results-and-analysis)
- [Contributing](#contributing)
- [License](#license)
- [Contact](#contact)

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

## Project Structure
