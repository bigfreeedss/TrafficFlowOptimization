import os
import traci
from sumo_env import SumoTrafficEnv

# -------------------
# SETTINGS
# -------------------
MAX_STEPS = 1800  # 30 minutes
GREEN_PHASE_DURATION = 15  # seconds per phase
USE_GUI = True

# -------------------
# LOAD ENVIRONMENT
# -------------------
env = SumoTrafficEnv(gui=USE_GUI)

# Reset SUMO
obs, _ = env.reset()
total_throughput = 0
total_wait_time = 0
current_phase_time = {tls: 0 for tls in env.tls_ids}
current_phases = {tls: 0 for tls in env.tls_ids}  # Start all at phase 0 (NS green)

print("🚦 Running Fixed-Time Traffic Control (15s per phase)...")

# -------------------
# SIMULATION LOOP
# -------------------
for step in range(MAX_STEPS):
    traci.simulationStep()

    # For every traffic light, keep fixed 15-second cycles
    for tls_id in env.tls_ids:
        current_phase_time[tls_id] += 1
        if current_phase_time[tls_id] >= GREEN_PHASE_DURATION:
            # Switch to opposite phase (0 → 2, 2 → 0)
            current_phases[tls_id] = (current_phases[tls_id] + 1) % 2
            traci.trafficlight.setPhase(tls_id, current_phases[tls_id] * 2)
            current_phase_time[tls_id] = 0

    # Collect statistics
    total_throughput += traci.simulation.getArrivedNumber()
    total_wait_time += sum(traci.edge.getWaitingTime(edge) for edge in env.edges)

env.close()

# -------------------
# RESULTS
# -------------------
print("\n📊 FIXED-TIME CONTROL RESULTS")
print(f"✅ Total Throughput: {total_throughput} vehicles")
print(f"✅ Average Wait Time per Step: {total_wait_time / MAX_STEPS:.2f} seconds")
print("🎯 Simulation Complete")
