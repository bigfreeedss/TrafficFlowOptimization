import os
import traci
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sumo_env import SumoTrafficEnv
import numpy as np


# -------------------------
# Custom Logging Callback
# -------------------------
class TrafficLoggingCallback(BaseCallback):
    def __init__(self, check_freq=5000, save_path="checkpoints", verbose=1):
        super(TrafficLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        os.makedirs(save_path, exist_ok=True)

    def _on_step(self) -> bool:
        # Every check_freq steps, log and save model
        if self.n_calls % self.check_freq == 0:
            avg_wait = 0
            avg_queue = 0
            if traci.isLoaded():
                edges = self.training_env.envs[0].edges
                avg_wait = np.mean([traci.edge.getWaitingTime(e) for e in edges])
                avg_queue = np.mean([traci.edge.getLastStepHaltingNumber(e) for e in edges])

            if self.verbose > 0:
                print(f"📊 Step {self.n_calls} | Avg Wait: {avg_wait:.2f} | Avg Queue: {avg_queue:.2f}")

            model_path = os.path.join(self.save_path, f"ppo_traffic_{self.n_calls}_steps.zip")
            self.model.save(model_path)
            print(f"💾 Saved model at {model_path}")

        return True


# -------------------------
# Training Script
# -------------------------
if __name__ == "__main__":
    # Create env
    env = DummyVecEnv([lambda: SumoTrafficEnv(gui=False)])

    # Define PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=0.0003,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,
        gamma=0.99,
    )

    # Callback for logging & saving
    callback = TrafficLoggingCallback(check_freq=10000, save_path="checkpoints")

    print("🚦 Starting training...")
    model.learn(total_timesteps=500_000, callback=callback)

    # Save final model
    model.save("ppo_traffic_final")
    print("✅ Training complete. Model saved as ppo_traffic_final.zip")

    env.close()
