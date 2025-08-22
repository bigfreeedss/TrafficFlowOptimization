# Traffic Flow Optimization Using Reinforcement Learning

##  Project Overview

Urban traffic congestion is a major global issue, causing economic loss, environmental damage, and reduced quality of life. Traditional fixed-time traffic light systems fail to adapt to dynamic traffic patterns.

This project explores **Reinforcement Learning (RL)** for **adaptive traffic signal control** using **Proximal Policy Optimization (PPO)**. Inspired by observations at the **Okponglo intersection in Accra**, we built a simulated **2x2 SUMO grid network** to test our model.

Our PPO agent achieved a **71.7% reduction in average waiting time** compared to a static baseline, providing strong proof-of-concept for RL-based traffic management in future smart cities.

---

## Methodology

### 1. Simulation Design

* **SUMO (Simulation of Urban Mobility):** Built a 2x2 grid with 4 intersections.
* **Traffic Flows:** Mixed heavy commuter flows, cross-traffic, and turning maneuvers.
* **Vehicle Types:** Cars and trucks with realistic acceleration/deceleration.

### 2. RL Environment

* **Observation Space:** Vehicle counts on all entry lanes.
* **Action Space:** MultiDiscrete (select traffic light phases for all 4 intersections).
* **Reward Function:**

  * Negative penalty for vehicle waiting time & queue length.
  * Positive reward for vehicles that exit the network.
  * Balanced to ensure **equity** across all approaches.

### 3. Training

* Algorithm: **PPO (Stable-Baselines3)**
* Timesteps: **500,000**
* Key Hyperparameters: `gamma=0.99`, `n_steps=2048`, `batch_size=64`, `ent_coef=0.01`, `lr=0.0003`.

---

##  Results

| Metric                | Baseline (Fixed-Time) | PPO Agent | Improvement |
| --------------------- | --------------------- | --------- | ----------- |
| Avg. Waiting Time (s) | 18.36s                | 2.60s     | 71.7% ↓     |
| Total Throughput      | 998 vehicles          | 1038      | +4.0% ↑     |
| Avg. Queue Length     | 18,321s               | 6,540s    | 64% ↓       |

 PPO agent dynamically extended green phases for heavy flows and shortened phases for empty approaches.
⚠️ Occasionally rapid phase-switching observed → future work should add a penalty for excessive switching.

---

##  Repository Structure

```bash
TrafficFlowOptimization/
├── network/
│   ├── grid.nod.xml
│   ├── grid.edg.xml
│   └── grid.net.xml
├── traffic/
│   ├── grid.rou.xml
│   ├── grid.sumocfg
│   └── tripinfo.xml
├── sumo_env.py             # Custom RL environment
├── train_agent.py          # PPO training script
├── evaluate_agent.py       # Evaluation script
├── avg_wait_time_script.py # Extract wait-time metrics
├── checkpoints/            # Saved PPO models
└── reports/
    └── TFO.pptx            # Final presentation
```

---

## Running the Project

### 1. Install Dependencies

```bash
pip install stable-baselines3 gymnasium torch
pip install sumolib traci
```

### 2. Train PPO Agent

```bash
python train_agent.py
```

### 3. Evaluate Model

```bash
python evaluate_agent.py
```

### 4. Extract Metrics

```bash
python avg_wait_time_script.py
```

---

##  Ethical Considerations

* **Equity:** Reward function penalizes long waits across **all lanes**, avoiding bias toward main roads.
* **Safety:** Real-world deployment must include **human override & fallback controls**.
* **Transparency:** Vehicle counts & logged rewards improve interpretability.
* **Efficiency:** Training is computationally costly, but deployed policies are lightweight.

---

## References

* Lopez, P. A., et al. (2018). *Microscopic traffic simulation using SUMO*. IEEE ITSC. [https://doi.org/10.1109/ITSC.2018.8569938](https://doi.org/10.1109/ITSC.2018.8569938)
* Schulman, J., et al. (2017). *Proximal Policy Optimization Algorithms*. arXiv. [https://doi.org/10.48550/arXiv.1707.06347](https://doi.org/10.48550/arXiv.1707.06347)
* Raffin, A., et al. (2021). *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR. [http://jmlr.org/papers/v22/20-1364.html](http://jmlr.org/papers/v22/20-1364.html)
* Genders, W., & Razavi, S. (2016). *Using a Deep RL Agent for Traffic Signal Control*. arXiv. [https://doi.org/10.48550/arXiv.1611.01142](https://doi.org/10.48550/arXiv.1611.01142)
* SUMO Documentation – [https://sumo.dlr.de/docs/](https://sumo.dlr.de/docs/)
* Gymnasium Documentation – [https://gymnasium.farama.org/](https://gymnasium.farama.org/)

---

✅ This README explains the **project motivation, methodology, reward design, results, repo structure, setup instructions, and ethical considerations** — so anyone who visits your GitHub can fully understand and replicate the work.

Do you want me to also add a **diagram/graph** (e.g., showing baseline vs PPO average wait time) inside the README for visualization?
