from src.env_creators import register
from tqdm import tqdm
from src.plotter import plotter
import os
from src.algorithm_config_generator import ALGORITHM, EXPERIMENT_NAME, TRAINING_ITERATIONS
from pprint import pprint

register()
path = os.path.join(os.getcwd(), 'results', 'plots', EXPERIMENT_NAME)
if not os.path.exists(path):
    print(
        f"Experiment {EXPERIMENT_NAME} does not exist. Creating new experiment...")
    os.makedirs(path)

TRAINING_RESULTS = []
checkpoint_dir = f'./results/checkpoints/{EXPERIMENT_NAME}'
print("Starting training...")
try:
    for i in tqdm(range(TRAINING_ITERATIONS), 'running_training_step'):
        result = ALGORITHM.train()
        pprint(result)
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
