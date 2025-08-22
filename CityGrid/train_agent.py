import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv
from sumo_env import SumoTrafficEnv

# -------------------
# SETTINGS
# -------------------
TIMESTEPS = 500_000        # Training steps
MODEL_DIR = "checkpoints"
MODEL_NAME = "ppo_traffic_optimized"

# Create checkpoint directory if it doesn't exist
os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------
# ENVIRONMENT SETUP
# -------------------
def make_env():
    # Use gui=False for faster training
    return SumoTrafficEnv(gui=False, min_green=10)

env = DummyVecEnv([make_env])

# -------------------
# MODEL SETUP
# -------------------
model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=0.0003,
    n_steps=2048,
    batch_size=64,
    n_epochs=10,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2
)

# -------------------
# CHECKPOINT CALLBACK
# -------------------
checkpoint_callback = CheckpointCallback(
    save_freq=50_000,
    save_path=MODEL_DIR,
    name_prefix=MODEL_NAME
)

# -------------------
# TRAINING
# -------------------
print("🚦 Starting Training...")
model.learn(
    total_timesteps=TIMESTEPS,
    callback=checkpoint_callback
)

# -------------------
# SAVE FINAL MODEL
# -------------------
final_model_path = os.path.join(MODEL_DIR, f"{MODEL_NAME}_final.zip")
model.save(final_model_path)
print(f"✅ Training Complete. Model saved at {final_model_path}")
