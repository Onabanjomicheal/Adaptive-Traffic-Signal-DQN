Optimizing Traffic Signals using Deep Q-Networks (DQN) in SUMO for Lagos Traffic
This repository contains the implementation of an intelligent traffic signal optimization system using Deep Q-Networks (DQN) within the SUMO (Simulation of Urban MObility) environment. The project aims to reduce traffic congestion and improve flow at intersections, specifically focusing on traffic patterns inspired by Lagos, Nigeria.

Table of Contents
Introduction

Features

Project Structure

Installation

Usage

Agents Implemented

Results and Analysis

Contributing

License

Contact

Introduction
Urban traffic congestion is a pervasive problem, particularly in rapidly growing cities like Lagos. Traditional fixed-time traffic signal systems often fail to adapt to dynamic traffic conditions, leading to inefficiencies. This project explores the application of Deep Reinforcement Learning (DRL), specifically Deep Q-Networks, to create adaptive traffic signal controllers that can learn optimal signal timings in real-time.

SUMO is used as the traffic simulation platform, providing a realistic environment to train and evaluate the DQN agent.

Features
Deep Q-Network (DQN) Agent: Implements a DQN model that learns optimal traffic light phases.

Fixed-Time Agent: A baseline comparison agent for evaluating performance.

Q-Learning Agent: Another reinforcement learning baseline for comparison.

SUMO Simulation Integration: Seamless interaction with SUMO for realistic traffic scenarios.

Data-Driven Optimization: The DQN agent learns from real-time traffic state (queue lengths, waiting times).

Performance Metrics: Tracks and visualizes key traffic metrics such as cumulative reward, queue length, and average waiting time.

Reproducible Environment: Project structure designed for easy setup and replication.

Project Structure
.
├── DQL_Agent.py                  # Deep Q-Learning agent implementation
├── FixedTime_Agent.py            # Fixed-Time traffic signal agent (baseline)
├── QLearning_Agent.py            # Q-Learning agent implementation (baseline)
├── README.md                     # Project documentation (this file)
├── .gitignore                    # Specifies intentionally untracked files to ignore
├── input data/                   # Contains raw input data files
│   ├── OSM data.xlsx             # OpenStreetMap data for network generation
│   └── Veh_Demand_Parameter.xlsx # Vehicle demand parameters
├── results/                      # Stores simulation outputs and plots
│   ├── Traffic metrics report.xlsx # Excel report of key traffic metrics
│   ├── Fixed Time_Queue Lenght.png # Plot: Queue length over time for Fixed-Time agent
│   ├── Fixed Timing_reward image.png # Plot: Reward over time for Fixed-Time agent
│   ├── DQN Plots/                # Directory for DQN agent's performance plots
│   │   └── ... (DQN specific plots like cumulative reward, queue length)
│   └── Q_Learning Plots/         # Directory for Q-Learning agent's performance plots
│       └── ... (Q-Learning specific plots)
├── osm.sumocfg                   # Main SUMO configuration file
├── osm.net.xml.gz                # Gzipped SUMO network file (referenced by osm.sumocfg)
├── osm_demand.rou.xml            # SUMO route file (generated/input, referenced by osm.sumocfg)
├── osm.add.xml                   # SUMO additional file (generated/input, referenced by osm.sumocfg)
├── osm.bus.rou.xml               # SUMO route file for buses (additional input)
├── osm.passenger.rou.xml         # SUMO route file for passengers (additional input)
├── osm.truck.rou.xml             # SUMO route file for trucks (additional input)
├── osm.vtypes.xml                # SUMO vehicle types definition (input)
├── osm.poly.xml                  # SUMO polygon definitions (input)
└── osm.stops.add                 # SUMO stops additions (input)

Installation
To set up the project locally, follow these steps:

Clone the repository:

git clone https://github.com/Onabanjomicheal/Adaptive-Traffic-Signal-DQN.git
cd Adaptive-Traffic-Signal-DQN

Install SUMO:
Ensure SUMO is installed on your system. You can download it from the SUMO website. Make sure the SUMO binaries are accessible in your system's PATH.

Create a Python virtual environment:

python -m venv .venv

Activate the virtual environment:

On Windows:

.venv\Scripts\activate

On macOS/Linux:

source .venv/bin/activate

Install required Python packages:

pip install sumolib pandas numpy tensorflow keras matplotlib

Usage
To run the traffic simulation with a specific agent:

Ensure SUMO is correctly installed and configured.

Activate your virtual environment (if not already active).

Run the desired agent script:

For Deep Q-Network Agent:

python DQL_Agent.py

For Fixed-Time Agent (Baseline):

python FixedTime_Agent.py

For Q-Learning Agent (Baseline):

python QLearning_Agent.py

The scripts will automatically launch the SUMO simulation and interact with it. Simulation outputs, including performance metrics and plots, will be saved in the results/ directory.

Agents Implemented
1. Deep Q-Network (DQN) Agent
This agent uses a neural network to approximate the Q-values for different traffic light phases. It learns to make decisions that maximize cumulative reward, typically defined by minimizing vehicle waiting times and queue lengths.

2. Fixed-Time Agent (Baseline)
A simple agent that applies predefined, static traffic light timings. This serves as a fundamental benchmark to evaluate the performance of adaptive agents.

3. Q-Learning Agent (Baseline)
A basic reinforcement learning agent that uses a Q-table to store and update Q-values. While less scalable than DQN for large state spaces, it provides a valuable comparison for the learning process.

Results and Analysis
After running the simulations, the results/ directory will contain:

Traffic metrics report.xlsx: A spreadsheet detailing various traffic flow metrics (e.g., average waiting time, total travel time) for each simulation run.

Plots for each agent (e.g., Fixed Time_Queue Lenght.png, Fixed Timing_reward image.png): Visualizations of key performance indicators over the simulation duration. These plots help in comparing the effectiveness of the different agents.

Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or bug fixes, please open an issue or submit a pull request.

License
This project is licensed under the MIT License. See the LICENSE file for details.
(Note: You might need to create a LICENSE file in your repository if you want to include it, though it's optional for now).

Contact
For any questions or inquiries, please contact:
[Your Name/Email/GitHub Profile Link]
