<!DOCTYPE html>
<html>
<head>
    <style>
        body {
            font-family: Arial, sans-serif;
        }
        .header {
            color: #2E86C1;
            font-size: 24px;
        }
        .subheader {
            color: #2874A6;
            font-size: 20px;
        }
        .code {
            background-color: #F4F6F6;
            padding: 10px;
            border-left: 3px solid #2E86C1;
            font-family: monospace;
        }
        .note {
            background-color: #F9E79F;
            padding: 10px;
            margin-top: 10px;
            font-size: 16px;
        }
        a {
            color: #3498DB;
        }
    </style>
</head>
<body>

    <div class="header">How To Use The Code</div>
    
    <div class="subheader">Setting Up the Environment</div>
    <p>Follow these steps to set up your environment:</p>
    <div class="code">
        python3 -m venv venv<br>
        source ./venv/bin/activate<br>
        pip install -r requirements.txt
    </div>

    <div class="subheader">Running the Code</div>
    <p>There are two entry points to the code: <code>train.py</code> and <code>run.py</code>.</p>
    <p><b>Starting with <code>train.py</code>:</b></p>
    <div class="code">
        python3 main.py --args...
    </div>
    <p>This will start the code with the algorithm and environment you supplied and will train the code.</p>

    <p><b>Arguments Explained:</b></p>
    <p>The possible arguments are:</p>
    <ul>
        <li><code>--algo</code>: The algorithm to use (default 'ppo').</li>
        <li><code>--env</code>: The environment for training (default 'box_pushing').</li>
        <li><code>--training_iterations</code>: Number of training iterations (default 2000).</li>
        <li><code>--gamma</code>: Discount factor for future rewards (default 0.95).</li>
        <li><code>--lr</code>: Learning rate (default 5e-5).</li>
        <li><code>--batch_size</code>: Size of the training batch (default 32).</li>
        <li><code>--epsilon</code>, <code>--epsilon_decay</code>, <code>--epsilon_min</code>: Parameters for exploration strategy.</li>
        <li><code>--framework</code>: The underlying framework (default 'torch').</li>
        <li><code>--seed</code>: Random seed (default 0).</li>
        <li><code>--log_level</code>: Logging level (default 'ERROR').</li>
    </ul>

    <div class="note">
        Note: The code snippet for parsing the arguments is not shown here, but you can understand each argument's possibilities from the provided code snippet.
    </div>

    <div class="subheader">Reviewing Results</div>
    <p>After running the code, check the <code>results</code> directory. It contains two subdirectories: <code>checkpoints</code> and <code>plots</code>. Metrics on the run can be found in <code>plots</code>, and the trained algorithm is saved under <code>results/plots||checkpoints/algo_env/algo_env_creation_date</code>.</p>

    <p><b>Using <code>run.py</code>:</b></p>
    <p>To load the trained algorithm and policies, modify the <code>run.py</code> file with the path to the checkpoint and the desired environment.</p>
    <div class="note">
        Note: The <code>run.py</code> file is just an example. Use it as a template to play around with and understand how to use the restored algorithm.
    </div>

    <div class="subheader">Further Documentation</div>
    <p>For more information on how to use the <code>run.py</code> file and algorithms, refer to the <a href="https://docs.ray.io/en/latest/rllib.html">RLlib documentation</a>. Consult the algorithms page in the documentation to understand which algorithms support multiagent training.</p>

</body>
</html>
