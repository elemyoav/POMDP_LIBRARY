from ray.rllib.algorithms import PPOConfig
from args import ARGS
from datetime import datetime

def get_config(name='ppo'):
    if name == 'ppo':
        return PPOConfig()
    else:
        raise NotImplementedError(f'config for {name} not implemented')

def build_with_env(algo_config, env_name):
    if env_name not in ['dec_tiger', 'tiger', 'dec_box_pushing', 'box_pushing']:
        raise NotImplementedError(f'env {env_name} not implemented')
    
    return algo_config.build(env=env_name)

def add_framework(config, framework='torch'):
    return config.framework(framework)

def add_debugging(config, seed=0, log_level='ERROR'):
    return config.debugging(seed=seed, log_level=log_level)

def add_rollouts(config, create_env_on_local_worker=True):
    return config.rollouts(create_env_on_local_worker=create_env_on_local_worker)

def add_training(config, gamma=0.9, lr=5e-5):
    return config.training(gamma=gamma, lr=lr)

def add_multi_agent(config, env_name, num_agents=2):
    if env_name.startswith('dec'):
        return config.multi_agent(
            policies=[f'policy_{i}' for i in range(num_agents)],
            policy_mapping_fn=lambda agent_id, _, **kwargs: f'policy_{agent_id[-1]}',
        )
    else:
        return config


alg_config = get_config(name=ARGS.algo)
alg_config = add_framework(alg_config, framework=ARGS.framework)
alg_config = add_debugging(alg_config, seed=ARGS.seed, log_level=ARGS.log_level)
alg_config = add_rollouts(alg_config)
alg_config = add_training(alg_config, gamma=ARGS.gamma, lr=ARGS.lr)
alg_config = add_multi_agent(alg_config, ARGS.env)

ALGORITHM = build_with_env(alg_config, ARGS.env)
EXPERIMENT_NAME = ARGS.algo + '_' + ARGS.env + '/' + ARGS.algo + '_' +ARGS.env + '_' + datetime.now().strftime("%Y%m%d-%H%M%S")
TRAINING_ITERATIONS = ARGS.training_iterations