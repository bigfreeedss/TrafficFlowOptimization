import os
import traci
from stable_baselines3 import PPO
from sumo_env import SumoTrafficEnv

# -------------------
# SETTINGS
# -------------------
MODEL_PATH = "ppo_traffic_final.zip"   # Path to your saved model
MAX_STEPS = 3000                       # How long to run the evaluation
USE_GUI = True                         # Always True for evaluation visualization

# -------------------
# CHECK MODEL
# -------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

# -------------------
# LOAD ENV + MODEL
# -------------------
env = SumoTrafficEnv(gui=USE_GUI, min_phase_time=10)  # match training setup
model = PPO.load(MODEL_PATH)

# -------------------
# RUN SIMULATION
# -------------------
obs, _ = env.reset()
total_reward = 0
total_throughput = 0
total_wait_time = 0

# Track per-junction stats
junction_waits = {tls: 0 for tls in env.tls_ids}

print("🚦 Starting Evaluation in SUMO GUI...")

for step in range(MAX_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    # log global metrics
    total_reward += reward
    total_throughput += traci.simulation.getArrivedNumber()
    total_wait_time += sum(traci.edge.getWaitingTime(edge) for edge in env.edges)

    # log per-junction wait time (sum of its incoming edges)
    for tls_id in env.tls_ids:
        incoming_edges = traci.trafficlight.getControlledLanes(tls_id)
        # getWaitingTime works on edges, but controlled lanes are lanes → take parent edge
        wait_time = sum(traci.lane.getWaitingTime(lane) for lane in incoming_edges)
        junction_waits[tls_id] += wait_time

    if terminated or truncated:
        break

env.close()

# -------------------
# RESULTS
# -------------------
print("\n📊 EVALUATION RESULTS")
print(f"✅ Total Reward: {total_reward:.2f}")
print(f"✅ Total Throughput: {total_throughput} vehicles")
print(f"✅ Average Wait Time per Step: {total_wait_time / (step+1):.2f} seconds")
print(f"✅ Simulation Steps: {step+1}")

print("\n📍 Per-Junction Average Wait Times:")
for tls_id, total_wait in junction_waits.items():
    avg_wait = total_wait / (step + 1)
    print(f"   - {tls_id}: {avg_wait:.2f} sec/step")

print("🎯 Evaluation Complete")
