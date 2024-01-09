# Interactive Mode README

## What is Interactive Mode?
Interactive Mode is an engaging feature that allows users to interact directly with the environments we've created. It offers a hands-on experience to see how episodes are played out in real-time, providing a unique and immersive way to understand and enjoy our environments.

## How to Use Interactive Mode
Running Interactive Mode is straightforward. Follow these simple steps:

1. **Open Your Command Line Interface**: Ensure you have command line access on your device.

2. **Run the Interactive Mode Command**:
   - Type the following commands:
     ```bash
     python3 -m venv venv
     ```

     ```bash
     source ./venv/bin/activate
     ```

     ```bash
     pip install -r requirements.txt
     ```
    
     ```
     python3 interactive/interactive.py --env <env-name> --num_games <number-of-games>
     ```
   - Replace `<env-name>` with the name of the environment you wish to play. 
     choose from 'dec_tiger', 'dec_box_pushing', 'dec_rock_sampling'
     
   - Replace `<number-of-games>` with the number of games you want to run.

3. **Enjoy Playing**: Once the command runs, the environment will launch in interactive mode, allowing you to play and explore.