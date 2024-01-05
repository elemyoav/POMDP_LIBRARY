from envs.box_pushing.box_pushing import DecBoxPushing


box_pushing_env = DecBoxPushing({})

box_pushing_env.reset()

print(box_pushing_env.step(
    {
        'agent_0': 2,
        'agent_1': 3
    }
))
