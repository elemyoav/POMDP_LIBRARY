# How to Run the Code

1. Once you have cloned the directory and are in VSCode, run:
   ```bash
   python3 -m venv venv
   
Activate the virtual environment:

bash

source ./venv/bin/activate

Install the requirements:

bash

pip install -r requirements.txt

Run the main script:

bash

    python3 ./main.py

If everything was set up correctly, you will see an insanely large JSON on the screen. This means everything went okay.
How to Write an Environment

To be able to implement an environment, you need to read the documentation: Ray RLlib Environments. This is simpler here but still uses a different interface.
How to Learn RLlib and Ray

There are 3 levels of seriousness you can take it to:
Minimum

Read the entire RLlib documentation: RLlib Docs. It's really important and short (about a 1-hour read). You will have a good enough understanding to make the code run. Make sure you focus on the Environment section since this is the only part we actually code in.
Medium

Read the entire Ray documentation: Ray Overview. This will take about 2-4 hours. It's the best option if you have the time, as we will be using other functionalities of this library, such as collecting metrics.
Extreme

Read an entire book on the library: Learning Ray (PDF). This is a big commitment, but you will become an expert in the library. The book is beginner-friendly and structured like a course. However, it's time-consuming.

You can also look at how I implemented the tiger environment in the env/tiger.py file, although it is not well documented yet and quite complex.
Plan

 Create and test box pushing and rock sampling (MultiAgentEnv only).
 Create a function to convert a MultiAgentEnv to a Gym Env.
 Learn how to collect metrics from a run.
 Run the experiments and collect all metrics.