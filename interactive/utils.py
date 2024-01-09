from args import ARGS
import os
import sys
from functools import reduce
sys.path.append(os.getcwd())

import envs



def get_env(env_name):
    if env_name == 'dec_tiger':
        return envs.DecTiger()
    elif env_name == 'dec_rock_sampling':
        return envs.DecRockSampling()
    elif env_name == 'dec_box_pushing':
        return envs.DecBoxPushing()
    else:
        raise ValueError("Invalid environment name")


ENV_NAME = ARGS.env
NUM_GAMES = ARGS.num_games
ENV = get_env(ENV_NAME)

def get_user_prompt(agent_id):
    if ENV_NAME.startswith('dec_'):
        available_actions = ENV.translator.get_all_actions()
    else:
        available_actions = ENV._env.translator.get_all_actions()

    actions_formatted = reduce(lambda x, y: x + y, [f'{i}: {action}\n' for i, action in enumerate(available_actions)])
    return f"""
Choose an action for {agent_id} from the following actions:
{actions_formatted}
\n"""

def get_user_input(agent_id):
    query = get_user_prompt(agent_id)
    return int(input(query))

def print_info(obs, reward, done):
    print(f"""
obs: {obs}
reward: {reward}
done: {done}
""")