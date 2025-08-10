import gymnasium as gym
import numpy as np
import traci
import os
from gymnasium import spaces


class SumoTrafficEnv(gym.Env):
    def __init__(self):
        super(SumoTrafficEnv, self).__init__()
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=100, shape=(8,), dtype=np.float32)

        # Path to the SUMO config file
        self.sumocfg = os.path.abspath(os.path.join("traffic", "grid.sumocfg"))

        # Valid edge IDs for state observation
        self.edges = [
            "entry_N1", "entry_N2",
            "entry_S1", "entry_S2",
            "entry_W1", "entry_W2",
            "entry_E1", "entry_E2"
        ]

        # Default traffic light ID (will update dynamically after reset)
        self.traffic_light_id = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if traci.isLoaded():
            traci.close()

        sumoBinary = os.path.join(os.environ["SUMO_HOME"], "bin", "sumo")
        traci.start([sumoBinary, "-c", self.sumocfg])

        # Discover the traffic light ID from the network
        traffic_lights = traci.trafficlight.getIDList()
        print("Available traffic lights:", traffic_lights)

        if not traffic_lights:
            raise RuntimeError("❌ No traffic lights found in the simulation.")
        
        self.traffic_light_id = traffic_lights[0]  # Use the first traffic light ID

        return self._get_state(), {}  # Return observation and info

    def step(self, action):
        self._apply_action(action)
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
        total_wait = sum([traci.edge.getWaitingTime(edge) for edge in self.edges])
        return -total_wait

    def _apply_action(self, action):
        if self.traffic_light_id is None:
            raise RuntimeError("❌ Traffic light ID is not set. Make sure reset() is called first.")
        
        if action == 0:
            traci.trafficlight.setPhase(self.traffic_light_id, 0)
        else:
            traci.trafficlight.setPhase(self.traffic_light_id, 2)

    def close(self):
        traci.close()
