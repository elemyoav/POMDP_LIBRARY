import ray
from ray.rllib.algorithms import qmix
from envs.tiger import DecTigerEnv

ray.init()
algo = qmix.QMix(env=DecTigerEnv, config={
    "env_config": {},  # config to pass to env class
})

while True:
    print(algo.train())
