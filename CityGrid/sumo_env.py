import gymnasium as gym
import numpy as np
import traci
import os
from gymnasium import spaces
import sumolib


class SumoTrafficEnv(gym.Env):
    def __init__(self, gui=False, min_phase_time=10):
        super(SumoTrafficEnv, self).__init__()
        self.gui = gui
        self.min_phase_time = min_phase_time
        self.step_count = 0
        self.current_phase_time = 0
        self.last_action = 0

        # Action space: 0 = keep current, 1 = NS green, 2 = EW green
        self.action_space = spaces.Discrete(3)

        # Observations: vehicle counts on 8 entry edges
        self.observation_space = spaces.Box(low=0, high=200, shape=(8,), dtype=np.float32)

        self.sumocfg = os.path.abspath(os.path.join("traffic", "grid.sumocfg"))

        # Load network once to get traffic lights & edges
        net = sumolib.net.readNet(os.path.abspath(os.path.join("network", "grid.net.xml")))
        self.tls_ids = [tl.getID() for tl in net.getTrafficLights()]
        print("🔗 Controlling traffic lights:", self.tls_ids)

        self.edges = [e.getID() for e in net.getEdges() if e.getID().startswith("entry")]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self.current_phase_time = 0
        self.last_action = 0

        if traci.isLoaded():
            traci.close()

        if self.gui:
            sumoBinary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui")
        else:
            sumoBinary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")

        traci.start([sumoBinary, "-c", self.sumocfg])
        return self._get_state(), {}

    def step(self, action):
        self.step_count += 1
        self.current_phase_time += 1

        # Only change if min green/red time passed
        if action != self.last_action and self.current_phase_time >= self.min_phase_time:
            self._apply_action(action)
            self.last_action = action
            self.current_phase_time = 0

        traci.simulationStep()
        state = self._get_state()
        reward = self._calculate_reward()
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = False

        return state, reward, terminated, truncated, {}

    def _get_state(self):
        return np.array([traci.edge.getLastStepVehicleNumber(e) for e in self.edges], dtype=np.float32)

    def _calculate_reward(self):
        wait_time = sum(traci.edge.getWaitingTime(e) for e in self.edges)
        throughput = traci.simulation.getArrivedNumber()
        queue_length = sum(traci.edge.getLastStepHaltingNumber(e) for e in self.edges)

        # Encourage throughput, discourage waiting & long queues
        return (0.5 * throughput) - (0.3 * wait_time) - (0.2 * queue_length)

    def _apply_action(self, action):
        for tls_id in self.tls_ids:
            if action == 1:  # NS green
                traci.trafficlight.setPhase(tls_id, 0)
            elif action == 2:  # EW green
                traci.trafficlight.setPhase(tls_id, 2)
            else:  # Keep current
                pass

    def close(self):
        if traci.isLoaded():
            traci.close()
