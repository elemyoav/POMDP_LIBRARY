import ray
from ray.rllib.algorithms import Algorithm
from envs.box_pushing.box_pushing import DecBoxPushing
from envs.team_problem_converter import convert_to_gym_env
import src.env_creators as env_creators

def run(checkpoint_path, env, num_episodes=1000):
    print('loading checkpoint')
    alg = Algorithm.from_checkpoint(checkpoint_path)

    for i in range(num_episodes):
        print('resetting env, episode: ', i)
        obs, _ = env.reset()
        print('initial obs is: ', obs)
        episode_reward = 0

        while True:
            print('computing action')
            action = alg.compute_actions(obs)
            print('action is: ', action)
            obs, reward, done, trunc, info = env.step(action)
            print('obs is: ', obs)
            print('reward is: ', reward)
            print('done is: ', done)
            print('trunc is: ', trunc)
            print('info is: ', info)
            episode_reward += reward
            if done or done['__all__']:
                break
            env.render()
        print(f'Episode {i} reward: {episode_reward}')

if __name__ == '__main__':
   env = convert_to_gym_env(DecBoxPushing())
   run('/home/elem/Desktop/POMDP_LIB/results/checkpoints/ppo_box_pushing/ppo_box_pushing_20240105-204452', env)
   