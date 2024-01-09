from ray.tune.registry import register_env
from envs.tiger.tiger import DecTigerEnv
from envs.box_pushing.box_pushing import DecBoxPushing
from envs.rock_sampling.rock_sampling import DecRockSampling
from envs.team_problem_converter import convert_to_gym_env



def register_dec_tiger(env_config):
    return DecTigerEnv(env_config)

def register_tiger(env_config):
    return convert_to_gym_env(DecTigerEnv(env_config))

def register_dec_box_pushing(env_config):
    return DecBoxPushing(env_config)

def register_box_pushing(env_config):
    return convert_to_gym_env(DecBoxPushing(env_config))

def register_dec_rock_sampling(env_config):
    return DecRockSampling(env_config)

def register_rock_sampling(env_config):
    return convert_to_gym_env(DecRockSampling(env_config))

def register():
    register_env('dec_tiger', register_dec_tiger)
    register_env('tiger', register_tiger)
    register_env('dec_box_pushing', register_dec_box_pushing)
    register_env('box_pushing', register_box_pushing)
    register_env('dec_rock_sampling', register_dec_rock_sampling)
    register_env('rock_sampling', register_rock_sampling)