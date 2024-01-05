import ray
import env_creators as env_creators
from algorithm_config_generator import ALGORITHM, EXPERIMENT_NAME, TRAINING_ITERATIONS
from visualize_training_results import plotter
import os

ray.init()
TRAINING_RESULTS = []
checkpoint_dir = f'./results/checkpoints/{EXPERIMENT_NAME}'
print("Starting training...")
try:
    for i in range(TRAINING_ITERATIONS):
        result = ALGORITHM.train()
        TRAINING_RESULTS.append(result)

except KeyboardInterrupt:
    pass



ALGORITHM.save(checkpoint_dir)
print(
    "An Algorithm checkpoint has been created inside directory: "
    f"'{checkpoint_dir}'.\n"
    "Individual Policy checkpoints can be found in "
    f"'{os.path.join(checkpoint_dir, 'policies')}'."
)

plotter(TRAINING_RESULTS)