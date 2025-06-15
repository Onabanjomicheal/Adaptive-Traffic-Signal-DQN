# 🚦 Lagos Traffic Optimization with Deep Q-Networks (DQN) in SUMO

![Isolo-Egbe Map](https://raw.githubusercontent.com/Onabanjomicheal/Adaptive-Traffic-Signal-DQN/main/isolo_egbe.png)

A Reinforcement Learning model that clears traffic in Lagos 59% faster than traditional systems—tested on real maps using SUMO and Deep Q-Networks.

## 🔥 Why This Matters
Lagos traffic isn’t just inconvenient—it’s catastrophic. The Oke Afa Roundabout and Isolo–Mushin Corridor are among the city's most congested intersections. Traditional fixed-timing systems collapse under pressure. This project uses Deep Reinforcement Learning to fix that.

Our DQN-based signal controller outperforms traditional systems, cutting travel time by over 30%, increasing speed by 50%, and completing nearly 2x more trips—all in a realistic SUMO simulation built from actual OSM map data.

## Table of Contents

- [Project Goals](#project_goals)
- [Agents Built](#agents_built)
- [How It Was Tested](#how_it_was_tested)
- [Agents Implemented](#agents-implemented)
- [Performance Highlights](#performance_highlights)
- [Demo Video](#demo_video)
- [Stack and Tools](#stack_tools)
- [Contributing](#contributing)
- [What to Do Next](#what_to_do_next)
- [Author](#author)

## 🎯 Project Goals
- Simulate Lagos traffic at key intersections (Oke Afa Roundabout, Isolo–Mushin) using SUMO.

- Train a Deep Q-Network agent to dynamically adapt traffic signals in real-time.

- Benchmark against Fixed-Time and classical Q-Learning agents.

- Measure speed, throughput, emissions, and travel efficiency.

## 🧠 Agents Built

| Agent                  | Description                                                                      |
| ---------------------- | -------------------------------------------------------------------------------- |
| ✅ **DQN Agent**        | Learns traffic light timing via neural networks. Maximizes flow, minimizes wait. |
| ⬜ **Q-Learning Agent** | Table-based RL. Simple but effective baseline.                                   |
| ❌ **Fixed-Time Agent** | Static signal schedule. Old-school, non-adaptive benchmark.                      |


## 🧪 How It Was Tested
- SUMO simulation (command-line, 0.5s step size).

- 3,600s runs for Fixed and QL. DQN terminated early due to full network clearance.

- Output data:

Trip info

Emissions (CO2, CO, HC, NOx, PMx, Fuel, Noise)

Speed, queue length, and travel time

Visuals and reports generated with Matplotlib, Excel, and Pandas.

## 📊 Performance Highlights

| Metric               | Fixed Timing | Q-Learning | DQN            |
| -------------------- | ------------ | ---------- | -------------- |
| **Trips Completed**  | 1290         | 1519       | **2401** ✅     |
| **Avg. Travel Time** | 550s         | 408s       | **361s** ✅     |
| **Avg. Speed**       | 4.15 m/s     | 5.21 m/s   | **6.31 m/s** ✅ |
| **Simulation Time**  | 3600s        | 3600s      | **1446s** ✅    |

DQN cleared all vehicles in under 25 minutes, while others needed a full hour—and still didn’t finish.

📁 See full breakdown in results/Traffic metrics report.xlsx

## 📽️ Demo Video
🚧 (To be added — consider recording QL or OSM sim)

Embed the video here once it's ready: .gif, .mp4, or YouTube iframe. Show it in action.
A basic reinforcement learning agent that uses a Q-table to store and update Q-values. While less scalable than DQN for large state spaces, it provides a valuable comparison for the learning process.

## 🧰 Stack and Tools
SUMO / NetEdit (network creation and traffic simulation)

Python: Core logic

TensorFlow + Keras (DQN Model)

SumoLib, Traci (interface with SUMO)

Pandas, Numpy, Matplotlib (data analysis and plotting)


## 🤝 Contributing
Got a better RL architecture? Want to simulate a new intersection in Lagos or Nairobi?
PRs and issues welcome.

## ✅ What to Do Next
Clone and run the sim.

Watch the demo (coming soon).

Try your own reward function.

Open an issue. Let's scale this.

## 👨‍💻 Author
- Onabanjo Micheal
- Passionate about AI for sustainable cities.
- 📫 Connect on LinkedIn
