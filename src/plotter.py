from src.algorithm_config_generator import EXPERIMENT_NAME
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st

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


def plot_episode_mean_with_confidence_intervals(results):

    results = list(map(lambda r: r['hist_stats']['episode_reward'], results))
    means = []
    confidence_intervals = []

    # Calculate mean and 95% confidence interval for each array
    for result in results:
        mean = np.mean(result)
        means.append(mean)

        # Calculate the standard error and the confidence interval
        se = st.sem(result)
        ci = st.t.interval(0.95, len(result)-1, loc=mean, scale=se)
        confidence_intervals.append(ci)

    plt.figure(figsize=(10, 5))

    # X-axis values (index of each array in 'results')
    x_values = list(range(len(results)))

    # Plot means
    plt.plot(x_values, means, 'o-', label='Mean')

    # Plot confidence intervals
    for x, (lower, upper) in zip(x_values, confidence_intervals):
        plt.plot([x, x], [lower, upper], color='r', marker='_')

    # Plot the means
    
    plt.xlabel('Array Index')
    plt.ylabel('Value')
    plt.title('Mean and Confidence Intervals of Arrays')
    plt.legend()
    plt.savefig(f'./results/plots/{EXPERIMENT_NAME}/avg_reward_per_episode_with_confidence_intervals.png')

    
def plotter(results):
    plot_episode_reward_mean(results)
    plot_episode_reward_min(results)
    plot_episode_reward_max(results)
    plot_episode_len_mean(results)
    plot_episode_mean_with_confidence_intervals(results)