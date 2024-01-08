from envs.box_pushing.box_pushing import DecBoxPushing
from envs.tiger.tiger import DecTigerEnv
from envs.rock_sampling.rock_sampling import DecRockSampling
from envs.team_problem_converter import convert_to_gym_env

env = convert_to_gym_env(DecTigerEnv())
env.render()
for i in range(3):
    done = False
    env.reset()
    while(not done):
        a0 = int(input("Choose action for agent 0: "))
        a1 = int(input("Choose action for agent 1: "))
        actions = {'agent_0': a0, 'agent_1': a1}
        obs, reward, done, _, _ = env.step(actions)
        env.render()