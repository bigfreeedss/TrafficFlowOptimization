import gymnasium as gym
import numpy as np
import traci
import os
from gymnasium import spaces
import sumolib


class SumoTrafficEnv(gym.Env):
    def __init__(self, gui=False):
        super(SumoTrafficEnv, self).__init__()
        self.gui = gui  

        # Path to SUMO config
        self.sumocfg = os.path.abspath(os.path.join("traffic", "grid.sumocfg"))

        # Discover all traffic lights automatically from the network
        net_path = os.path.abspath(os.path.join("network", "grid.net.xml"))
        net = sumolib.net.readNet(net_path)
        self.tls_ids = [tls.getID() for tls in net.getTrafficLights()]
        print("🔗 Controlling traffic lights:", self.tls_ids)

        # Action space: choose a phase for each traffic light
        self.action_space = spaces.MultiDiscrete([
            traci.trafficlight.getPhaseNumber(tls_id)
            if traci.isLoaded() else 2  # fallback before sim starts
            for tls_id in self.tls_ids
        ])

        # Observation space: vehicle counts on key edges
        self.edges = [
            "entry_N1", "entry_N2",
            "entry_S1", "entry_S2",
            "entry_W1", "entry_W2",
            "entry_E1", "entry_E2"
        ]
        self.observation_space = spaces.Box(
            low=0, high=1000, shape=(len(self.edges),), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if traci.isLoaded():
            traci.close()

        sumoBinary = os.path.join(
            os.environ["SUMO_HOME"], "bin", "sumo-gui" if self.gui else "sumo"
        )
        traci.start([sumoBinary, "-c", self.sumocfg])

        # Re-check how many phases each traffic light actually has
        self.action_space = spaces.MultiDiscrete([
            traci.trafficlight.getPhaseNumber(tls_id)
            for tls_id in self.tls_ids
        ])

        return self._get_state(), {}

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
        """ Reward = -waiting time - queue + throughput bonus """
        total_wait = sum(traci.edge.getWaitingTime(edge) for edge in self.edges)
        queue_length = sum(traci.edge.getLastStepHaltingNumber(edge) for edge in self.edges)
        throughput = traci.simulation.getArrivedNumber()

        # STRONGER weights against congestion
        reward = - (1.0 * total_wait + 0.7 * queue_length) + (0.5 * throughput)
        return reward

    def _apply_action(self, action):
        # Ensure action is iterable (for single-light cases)
        if isinstance(action, (int, np.integer)):
            action = [action]

        for tls_id, phase in zip(self.tls_ids, action):
            traci.trafficlight.setPhase(tls_id, int(phase))

    def close(self):
        traci.close()
