from ray.tune.registry import register_env
from envs.tiger.tiger import DecTigerEnv
from envs.team_problem_converter import convert_to_gym_env



def register_dec_tiger(env_config):
    return DecTigerEnv(env_config)

def register_tiger(env_config):
    return convert_to_gym_env(DecTigerEnv(env_config))

register_env('dec_tiger', register_dec_tiger)
register_env('tiger', register_tiger)