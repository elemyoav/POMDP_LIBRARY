import ray
from ray.rllib.algorithms.ppo import PPOConfig
from envs.tiger import DecTigerEnv
from ray.tune.registry import get_trainable_cls
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime


EXPERIMENT_NAME = 'PPO_Tiger' + datetime.now().strftime("%Y%m%d-%H%M%S")
ray.init()

ppo_config = (
    PPOConfig()\
    .framework('torch')\
    .rollouts(create_env_on_local_worker=True)\
    .debugging(seed=0, log_level='ERROR')
)

ppo_config = ppo_config.multi_agent(
    policies=['policy_0', 'policy_1'],
    policy_mapping_fn=lambda agent_id, _, **kwargs: 'policy_0' if agent_id == 'agent_0' else 'policy_1',
)

ppo = ppo_config.build(env=DecTigerEnv)
rewards1 = []
rewards2 = []

for i in range(3):
    result = ppo.train()
    rewards1.append(result['policy_reward_mean']['policy_0'])
    rewards2.append(result['policy_reward_mean']['policy_1'])
    print('episode_reward_mean: ', result['episode_reward_mean'])


ppo.save(f'./results/checkpoints/{EXPERIMENT_NAME}')
plt.figure(figsize=(4,3))
plt.plot(rewards1, label='agent_0')
plt.plot(rewards2, label='agent_1')
plt.plot(np.array(rewards1) + np.array(rewards2), label='total', linewidth=3, color='black')
plt.xlabel('iteration')
plt.ylabel('reward')

plt.savefig(f'./results/plots/{EXPERIMENT_NAME}.png')
plt.show()