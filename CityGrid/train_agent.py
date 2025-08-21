import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sumo_env import SumoTrafficEnv

# -------------------
# SETTINGS
# -------------------
TOTAL_TIMESTEPS = 500_000        # train steps (adjust if too long)
CHECKPOINT_DIR = "checkpoints"
MODEL_PATH = "ppo_traffic_final.zip"
USE_GUI = False                  # turn ON for debugging, OFF for faster training

# -------------------
# CREATE ENV
# -------------------
def make_env():
    return SumoTrafficEnv(gui=USE_GUI, min_phase_time=10)

env = DummyVecEnv([make_env])

# -------------------
# CALLBACKS
# -------------------
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,   # save every 50k steps
    save_path=CHECKPOINT_DIR,
    name_prefix="ppo_traffic"
)

# -------------------
# TRAINING
# -------------------
print("🚦 Starting PPO Training...")
model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=1024,
    batch_size=64,
    ent_coef=0.01,   # encourages exploration
    gamma=0.99
)

model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)
model.save(MODEL_PATH)
print(f"✅ Training complete. Model saved at {MODEL_PATH}")

# -------------------
# QUICK TEST (OPTIONAL)
# -------------------
print("▶ Running quick evaluation after training...")
test_env = SumoTrafficEnv(gui=True, min_phase_time=10)  # open SUMO GUI
obs, _ = test_env.reset()
total_reward = 0

for step in range(1000):  # run 1000 steps to see behavior
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = test_env.step(action)
    total_reward += reward
    if terminated or truncated:
        break

test_env.close()
print(f"🎯 Quick Test Finished | Total Reward: {total_reward:.2f}")
