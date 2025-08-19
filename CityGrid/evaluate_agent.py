import os
import traci
from stable_baselines3 import PPO
from sumo_env import SumoTrafficEnv

# -------------------------
# SETTINGS
# -------------------------
MODEL_PATH = "ppo_traffic_final.zip"   # final trained model
MAX_STEPS = 3000                       # number of simulation steps
USE_GUI = True                         # show SUMO GUI

# -------------------------
# CHECK MODEL FILE
# -------------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

# -------------------------
# ENVIRONMENT
# -------------------------
env = SumoTrafficEnv(gui=USE_GUI)

# -------------------------
# LOAD MODEL
# -------------------------
print(f"📂 Loading trained model from {MODEL_PATH}...")
model = PPO.load(MODEL_PATH)

# -------------------------
# EVALUATION
# -------------------------
obs, _ = env.reset()
total_reward = 0
total_throughput = 0
total_wait_time = 0

print("🚦 Starting evaluation in SUMO GUI...")

for step in range(MAX_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    total_reward += reward
    total_throughput += traci.simulation.getArrivedNumber()
    total_wait_time += sum(traci.edge.getWaitingTime(edge) for edge in env.edges)

    if terminated or truncated:
        break

env.close()

# -------------------------
# RESULTS
# -------------------------
print("\n📊 EVALUATION RESULTS")
print(f"Total Reward: {total_reward:.2f}")
print(f"Total Throughput: {total_throughput}")
print(f"Average Wait Time per Step: {total_wait_time / (step+1):.2f} seconds")
print(f"Simulation Steps: {step+1}")
print("✅ Evaluation Complete")
