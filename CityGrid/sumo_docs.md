# Team 9: SUMO City Grid Simulation Environment

## 1. Introduction

This document provides a detailed overview of the simulation environment created for the Traffic Flow Optimization project. The environment, built using SUMO (Simulation of Urban MObility), serves as the virtual world for training and evaluating our Reinforcement Learning agent. The goal was to design a network that is complex enough to be a meaningful challenge for an AI controller, yet simple enough to be understandable and computationally efficient.

The simulation consists of a 2x2 city grid with multi-lane roads, four intelligent traffic light intersections, and a diverse set of traffic flows designed to mimic real-world urban conditions.

## 2. Network Architecture

The road network (`grid.net.xml`) forms the static foundation of our city. It was procedurally generated from raw node and edge files (`grid.nod.xml`, `grid.edg.xml`) using SUMO's `netconvert` tool.

### 2.1. Intersections (Nodes)

The grid contains a total of **12 nodes**:

- **4 Central Intersections:** These are the core of the grid, labeled `c1_1`, `c1_2`, `c2_1`, and `c2_2`. Each of these is a `traffic_light` type node, meaning their traffic flow is controlled by a signal program. These are the intersections our AI agent will control.
- **8 Peripheral Nodes:** These nodes (`N1`, `N2`, `S1`, `S2`, `W1`, `W2`, `E1`, `E2`) act as the entry and exit points for all traffic into and out of the city grid.

### 2.2. Roads (Edges)

All roads in the simulation are **two-lane**, allowing for more complex traffic dynamics like overtaking and lane choice. The roads connect the nodes to form the grid structure, with distinct edges for each direction of travel (e.g., the road from `c1_1` to `c1_2` is separate from the road from `c1_2` to `c1_1`).

## 3. Traffic Flow Dynamics

The traffic itself (`grid.rou.xml`) is designed to create challenging, non-uniform scenarios for the AI to solve. Instead of an even distribution, we have defined specific, realistic traffic patterns using a combination of vehicle types and routes.

### 3.1. Vehicle Types

Two types of vehicles are present in the simulation to add variety in acceleration and speed:

- **Cars (`vType="car"`):** Standard passenger vehicles with high acceleration and a maximum speed of 20 m/s. They are colored yellow.
- **Trucks (`vType="truck"`):** Heavier vehicles with slower acceleration and a lower top speed of 15 m/s. They are colored green.

### 3.2. Defined Routes and Flows

We have established four primary traffic flows to simulate a typical morning commute scenario:

1.  **Heavy Commuter Traffic (West to East):** The dominant flow in the simulation (`flow_W1_E1`) consists of a high volume of cars (one every 4 seconds) traveling straight through the northern part of the grid. This represents the main arterial route.
2.  **Moderate Commercial Traffic (North to South):** A flow of trucks (`flow_N1_S1`) crosses the main commuter path, appearing every 10 seconds. This creates a consistent point of conflict that the AI must manage.
3.  **Lighter Opposing Traffic (East to West):** A smaller flow of cars (`flow_E2_W2`) travels through the southern part of the grid (one every 7 seconds), ensuring the AI must balance demands across the entire network.
4.  **Complex Turning Traffic:** A dedicated flow of cars (`flow_left_turn_N1_W2`) is defined to make a challenging left turn across the grid (one every 12 seconds), forcing the AI to create safe gaps in oncoming traffic.

This combination of intersecting heavy, moderate, and turning traffic flows ensures that a simple, fixed-timer traffic light system will perform poorly, creating a clear opportunity for our AI agent to demonstrate significant improvement.
