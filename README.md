# How to Use the Code

## Getting Started

### Step 1: Download All Necessary Dependencies
To begin, set up your environment and install required packages:

1. Create and activate a virtual environment:
   - python3 -m venv venv
   - source ./venv/bin/activate

2. Install dependencies from the requirements file:
   - pip install -r requirements.txt

## How to Train an Agent

### Step 2: Configure Your Training Environment
For training setup:

1. Visit the config directory (link provided) and choose your desired environment and algorithm based on the YAML file.
2. Adjust the values in the YAML file as needed.

### Step 3: Execute Training
To train your agent:

- Run the command:
  - rllib train file /path/to/your/file.yaml

Note: This command trains the algorithm as per your configuration. Post-training, the algorithm will be saved in the ~/ray_results directory. The terminal's stdout will display the full path.

## How to Visualize Training Results

### Step 4: Visualize Outcomes
For result visualization:

- Execute the command:
  - tensorboard --logdir ~/ray_results/<experiment_name>

- Then, open your web browser and navigate to localhost:6006 to view the run metrics.

Hint: The complete command will be part of the stdout from the train command.

## How to Change Environment Settings

### Step 5: Modify Environment Settings
For environment adjustments:

1. Navigate to the env directory (link provided) and select the environment you wish to modify.
2. To alter rewards, open the rewards.py file and adjust the values as needed.