# How To Use The Code

## Setting Up the Environment

Once you cloned th repository on your local machine cd into it and run the following commands in the terminal:

```bash
python3 -m venv venv
```
```bash
source ./venv/bin/activate
```
```bash
pip install -r requirements.txt
```

## Running the Code

There are two entry points to the code: `train.py` and `run.py`.

### Starting with `train.py`:

Run the following command:

```bash
python3 train.py --args...
```

This will start the code with the algorithm and environment you supplied and will train the code.

### Arguments Explained:

The possible arguments are:

- `--algo`: The algorithm to use (default 'ppo', choose from 'ppo', 'marwil', 'appo', 'impala', 'bcc', 'cql', 'dqn', 'sacc').
- `--env`: The environment for training (default 'box_pushing', choose from 'dec_box_pushing', 'box_pushing', 'dec_tiger', 'tiger').
- `--training_iterations`: Number of training iterations (default 2000).
- `--gamma`: Discount factor for future rewards (default 0.95).
- `--lr`: Learning rate (default 5e-5).
- `--batch_size`: Size of the training batch (default 32).
- `--epsilon`, `--epsilon_decay`, `--epsilon_min`: Parameters for exploration strategy.
- `--framework`: The underlying framework (default 'torch', possibilities are 'torch', 'tf', 'tf2').
- `--seed`: Random seed (default 0).
- `--log_level`: Logging level (default 'ERROR').


## Reviewing Results

After running the code, check the `results` directory. It contains two subdirectories: `checkpoints` and `plots`. Metrics on the run can be found in `plots`, and the trained algorithm is saved under `results/plots||checkpoints/algo_env/algo_env_creation_date`.

### Using `run.py`:

To load the trained algorithm and policies, modify the `run.py` file with the path to the checkpoint and the desired environment.

> **Note**: The `run.py` file is just an example. Use it as a template to play around with and understand how to use the restored algorithm.

## Further Documentation

For more information on how to use the `run.py` file and algorithms, refer to the [RLlib documentation](https://docs.ray.io/en/master/rllib/index.html). Consult the algorithms page in the documentation to understand which algorithms support multiagent training.

if you want to learn how to use rllib please try this [RLlib Course](https://applied-rl-course.netlify.app/)
