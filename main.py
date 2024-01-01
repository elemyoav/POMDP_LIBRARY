import gymnasium as gym
import ray
from ray.rllib.algorithms import ppo
from envs.tiger import DecTigerEnv

ray.init()
algo = ppo.PPO(env=DecTigerEnv, config={
    "env_config": {},  # config to pass to env class
})

while True:
    print(algo.train())
