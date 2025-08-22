import gymnasium as gym
import numpy as np
import traci
import os
from gymnasium import spaces
import sumolib

class SumoTrafficEnv(gym.Env):
    def __init__(self, gui=False, min_green=10):
        super(SumoTrafficEnv, self).__init__()
        self.gui = gui
        self.min_green = min_green  # minimum time before switching lights
        self.action_space = spaces.MultiDiscrete([2] * len(self._get_tls_ids()))
        self.observation_space = spaces.Box(low=0, high=100, shape=(len(self._get_edge_ids()),), dtype=np.float32)

        self.sumocfg = os.path.abspath(os.path.join("traffic", "grid.sumocfg"))
        self.edges = self._get_edge_ids()
        self.tls_ids = self._get_tls_ids()
        self.tls_last_switch = {tls: 0 for tls in self.tls_ids}

    def _get_edge_ids(self):
        net_path = os.path.abspath(os.path.join("network", "grid.net.xml"))
        net = sumolib.net.readNet(net_path)
        return [e.getID() for e in net.getEdges() if e.getID() != ":"]

    def _get_tls_ids(self):
        net_path = os.path.abspath(os.path.join("network", "grid.net.xml"))
        net = sumolib.net.readNet(net_path)
        return [tls.getID() for tls in net.getTrafficLights()]

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if traci.isLoaded():
            traci.close()

        sumo_binary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo-gui" if self.gui else "sumo")
        traci.start([sumo_binary, "-c", self.sumocfg])

        self.tls_last_switch = {tls: 0 for tls in self.tls_ids}
        self.current_step = 0
        return self._get_state(), {}

    def step(self, action):
        self.current_step += 1

        for i, tls_id in enumerate(self.tls_ids):
            if self.current_step - self.tls_last_switch[tls_id] >= self.min_green:
                traci.trafficlight.setPhase(tls_id, int(action[i]) * 2)
                self.tls_last_switch[tls_id] = self.current_step

        traci.simulationStep()

        state = self._get_state()
        reward = self._calculate_reward()
        terminated = traci.simulation.getMinExpectedNumber() <= 0
        truncated = False
        return state, reward, terminated, truncated, {}

    def _get_state(self):
        return np.array(
            [traci.edge.getLastStepVehicleNumber(edge) for edge in self.edges],
            dtype=np.float32
        )

    def _calculate_reward(self):
        total_wait = sum(traci.edge.getWaitingTime(edge) for edge in self.edges)
        queue_length = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in self.edges)
        throughput = traci.simulation.getArrivedNumber()
        switch_penalty = sum([1 for tls in self.tls_ids if self.current_step - self.tls_last_switch[tls] < self.min_green])

        return - (0.5 * total_wait + 0.3 * queue_length + 0.2 * switch_penalty) + (0.6 * throughput)

    def close(self):
        traci.close()
