# How To Use The Code

## Setting Up the Environment

Follow these steps to set up your environment:

`bash
python3 -m venv venv
source ./venv/bin/activate
pip install -r requirements.txt
`

## Running the Code

There are two entry points to the code: `train.py` and `run.py`.

### Starting with `train.py`:

Run the following command:

`bash
python3 main.py --args...
`

This will start the code with the algorithm and environment you supplied and will train the code.

### Arguments Explained:

The possible arguments are:

- `--algo`: The algorithm to use (default 'ppo').
- `--env`: The environment for training (default 'box_pushing').
- `--training_iterations`: Number of training iterations (default 2000).
- `--gamma`: Discount factor for future rewards (default 0.95).
- `--lr`: Learning rate (default 5e-5).
- `--batch_size`: Size of the training batch (default 32).
- `--epsilon`, `--epsilon_decay`, `--epsilon_min`: Parameters for exploration strategy.
- `--framework`: The underlying framework (default 'torch').
- `--seed`: Random seed (default 0).
- `--log_level`: Logging level (default 'ERROR').

> **Note**: The code snippet for parsing the arguments is not shown here, but you can understand each argument's possibilities from the provided code snippet.

## Reviewing Results

After running the code, check the `results` directory. It contains two subdirectories: `checkpoints` and `plots`. Metrics on the run can be found in `plots`, and the trained algorithm is saved under `results/plots||checkpoints/algo_env/algo_env_creation_date`.

### Using `run.py`:

To load the trained algorithm and policies, modify the `run.py` file with the path to the checkpoint and the desired environment.

> **Note**: The `run.py` file is just an example. Use it as a template to play around with and understand how to use the restored algorithm.

## Further Documentation

For more information on how to use the `run.py` file and algorithms, refer to the [RLlib documentation](https://docs.ray.io/en/master/rllib/index.html). Consult the algorithms page in the documentation to understand which algorithms support multiagent training.

if you want to learn how to use rllib please try this [RLlib Course](https://applied-rl-course.netlify.app/)
