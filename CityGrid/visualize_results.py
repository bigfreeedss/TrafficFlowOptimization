import pandas as pd
import matplotlib.pyplot as plt

# Load results
data = pd.read_csv("evaluation_results.csv")

# Preview first few rows
print("\n📄 Preview of Results:")
print(data.head())

# -------------------
# Plot 1: Cumulative Reward
# -------------------
plt.figure(figsize=(8, 5))
plt.plot(data["step"], data["cumulative_reward"], label="Cumulative Reward")
plt.xlabel("Step")
plt.ylabel("Cumulative Reward")
plt.title("Cumulative Reward over Simulation")
plt.legend()
plt.grid(True)
plt.show()

# -------------------
# Plot 2: Throughput per Step
# -------------------
plt.figure(figsize=(8, 5))
plt.plot(data["step"], data["throughput"], color="green", label="Throughput per Step")
plt.xlabel("Step")
plt.ylabel("Throughput (vehicles)")
plt.title("Vehicle Throughput per Step")
plt.legend()
plt.grid(True)
plt.show()

# -------------------
# Plot 3: Average Wait Time
# -------------------
plt.figure(figsize=(8, 5))
plt.plot(data["step"], data["avg_wait_time"], color="red", label="Average Wait Time")
plt.xlabel("Step")
plt.ylabel("Average Wait Time (seconds)")
plt.title("Average Wait Time per Step")
plt.legend()
plt.grid(True)
plt.show()
