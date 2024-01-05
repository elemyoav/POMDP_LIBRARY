from algorithm_config_generator import EXPERIMENT_NAME
import matplotlib.pyplot as plt


def plot_episode_reward_mean(results):
    plt.figure(figsize=(10, 5))
    plt.plot([r['episode_reward_mean'] for r in results])
    plt.xlabel('Iteration')
    plt.ylabel('Avg Reward per Episode')
    plt.title(f'Avg Reward per Episode over {len(results)} Iterations')
    plt.savefig(f'./results/plots/{EXPERIMENT_NAME}/avg_reward_per_episode.png')

def plot_episode_reward_min(results):
    plt.figure(figsize=(10, 5))
    plt.plot([r['episode_reward_min'] for r in results])
    plt.xlabel('Iteration')
    plt.ylabel('Min Reward per Episode')
    plt.title(f'Min Reward per Episode over {len(results)} Iterations')
    plt.savefig(f'./results/plots/{EXPERIMENT_NAME}/min_reward_per_episode.png')

def plot_episode_reward_max(results):
    plt.figure(figsize=(10, 5))
    plt.plot([r['episode_reward_max'] for r in results])
    plt.xlabel('Iteration')
    plt.ylabel('Max Reward per Episode')
    plt.title(f'Max Reward per Episode over {len(results)} Iterations')
    plt.savefig(f'./results/plots/{EXPERIMENT_NAME}/max_reward_per_episode.png')

def plot_episode_len_mean(results):
    plt.figure(figsize=(10, 5))
    plt.plot([r['episode_len_mean'] for r in results])
    plt.xlabel('Iteration')
    plt.ylabel('Avg Episode Length')
    plt.title(f'Avg Episode Length over {len(results)} Iterations')
    plt.savefig(f'./results/plots/{EXPERIMENT_NAME}/avg_episode_length.png')


def plotter(results):
    plot_episode_reward_mean(results)
    plot_episode_reward_min(results)
    plot_episode_reward_max(results)
    plot_episode_len_mean(results)