import ray
from ray.rllib.algorithms import Algorithm
from envs.tiger.tiger import DecTigerEnv

def run(checkpoint_path, env, num_episodes=1000):
    alg = Algorithm.from_checkpoint(checkpoint_path)

    obs, _ = env.reset()
    for i in range(num_episodes):
        episode_reward = 0
        while True:
            action = alg.compute_action(obs)
            obs, reward, done, trunc, info = env.step(action)
            episode_reward += reward
            if done or done['__all__']:
                break
            env.render()
        print(f'Episode {i} reward: {episode_reward}')

if __name__ == '__main__':
   ray.init()
   env = DecTigerEnv({})
   run('./results/checkpoints/ppo_dec_tiger_20201208-143143', env)
   