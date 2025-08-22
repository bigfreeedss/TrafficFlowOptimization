import os
import csv
import traci
from stable_baselines3 import PPO
from sumo_env import SumoTrafficEnv

# -------------------
# SETTINGS
# -------------------
MODEL_PATH = "checkpoints/ppo_traffic_optimized_final.zip"  # Trained model path
RESULTS_CSV = "evaluation_results.csv"                      # File to store results
MAX_STEPS = 3000                                            # Simulation steps

# -------------------
# VALIDATE MODEL FILE
# -------------------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"❌ Model file not found: {MODEL_PATH}")

# -------------------
# LOAD ENVIRONMENT IN GUI MODE
# -------------------
env = SumoTrafficEnv(gui=True, min_green=10)  # Smooth simulation with GUI
model = PPO.load(MODEL_PATH)

# -------------------
# RESET SIMULATION
# -------------------
obs, _ = env.reset()
total_reward = 0
total_throughput = 0
total_wait_time = 0

# Prepare data list for CSV
data_log = []

print("🚦 Starting Evaluation in SUMO GUI...")

# -------------------
# RUN SIMULATION
# -------------------
for step in range(MAX_STEPS):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)

    total_reward += reward
    step_throughput = traci.simulation.getArrivedNumber()
    step_wait = sum(traci.edge.getWaitingTime(edge) for edge in env.edges)
    total_throughput += step_throughput
    total_wait_time += step_wait

    # Log step data
    data_log.append({
        "step": step + 1,
        "reward": reward,
        "cumulative_reward": total_reward,
        "throughput": step_throughput,
        "cumulative_throughput": total_throughput,
        "avg_wait_time": total_wait_time / (step + 1)
    })

    # End simulation if no more cars
    if terminated or truncated:
        break

# -------------------
# CLOSE SIMULATION
# -------------------
env.close()

# -------------------
# SAVE RESULTS TO CSV
# -------------------
with open(RESULTS_CSV, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=[
        "step",
        "reward",
        "cumulative_reward",
        "throughput",
        "cumulative_throughput",
        "avg_wait_time"
    ])
    writer.writeheader()
    writer.writerows(data_log)

# -------------------
# PRINT SUMMARY
# -------------------
print("\n📊 EVALUATION RESULTS")
print(f"✅ Total Reward: {total_reward:.2f}")
print(f"✅ Total Throughput: {total_throughput} vehicles")
print(f"✅ Average Wait Time per vehicle: {total_wait_time / (step+1):.2f} seconds")
print(f"✅ Simulation Steps: {step+1}")
print(f"📁 Results saved to: {RESULTS_CSV}")
print("🎯 Evaluation Complete")
