import ray
import env_creators as env_creators
from algorithm_config_generator import ALGORITHM, EXPERIMENT_NAME, TRAINING_ITERATIONS
from visualize_training_results import plot_results

ray.init()
TRAINING_RESULTS = []
print("Starting training...")
try:
    for i in range(TRAINING_ITERATIONS):
        result = ALGORITHM.train()
        TRAINING_RESULTS.append(result)

except KeyboardInterrupt:
    pass


ALGORITHM.save(f'./results/checkpoints/{EXPERIMENT_NAME}')


print("Done!")
print("Evaluating the algorithm...")
print("After running an evaluation of the algorithm, got an avg_reward_per_episode of:"\
      ,ALGORITHM.evaluate()["evaluation"]["episode_reward_mean"])

plot_results()