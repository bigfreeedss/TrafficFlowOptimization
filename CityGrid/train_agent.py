from stable_baselines3 import PPO
from sumo_env import SumoTrafficEnv
from stable_baselines3.common.env_checker import check_env

env = SumoTrafficEnv()

check_env(env)

model = PPO("MlpPolicy", env, verbose=1)
model.learn(total_timesteps=10000)
model.save("ppo_traffic_model")
env.close()
