import ray
from ray.rllib.algorithms.ppo import PPOConfig
from envs.tiger.tiger import DecTigerEnv
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
    .environment(DecTigerEnv)
)

ppo_config = ppo_config.multi_agent(
    policies=['policy_0', 'policy_1'],
    policy_mapping_fn=lambda agent_id, _, **kwargs: 'policy_0' if agent_id == 'agent_0' else 'policy_1',
)

ppo = ppo_config.build()
rewards1 = []
rewards2 = []

print("Starting training...")
for i in range(100):
    result = ppo.train()
    rewards1.append(result['policy_reward_mean']['policy_0'])
    rewards2.append(result['policy_reward_mean']['policy_1'])
    print('episode_reward_mean: ', result['episode_reward_mean'])


ppo.save(f'./results/checkpoints/{EXPERIMENT_NAME}')


print("Done!")
print("Evaluating the algorithm...")
print("After running an evaluation of the algorithm, got an avg_reward_per_episode of:"\
      ,ppo.evaluate()["evaluation"]["episode_reward_mean"])

plt.figure(figsize=(4,3))
plt.plot(rewards1, label='agent_0')
plt.plot(rewards2, label='agent_1')
plt.plot(np.array(rewards1) + np.array(rewards2), label='total', linewidth=3, color='black')
plt.xlabel('iteration')
plt.ylabel('reward')

plt.savefig(f'./results/plots/{EXPERIMENT_NAME}.png')
plt.show()