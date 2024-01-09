from utils import get_user_input, print_info, ENV, NUM_GAMES

def main():
    """
    This function will run a gam of the chosen environment in the terminal and allow the user to control the agents.
    To activate this function run the following command:
    python interactive/interactive.py --env <env_name> --num_games <num_games>
    the options for env_name are:
    - dec_tiger
    - dec_rock_sampling
    - dec_box_pushing

    the default option is dec_tiger

    the options for num_games are any positive integer
    the default option is 3

    in each game the user will be prompted to choose an action for each agent
    the actions will be listed in the terminal, and the user will be prompted 
    to enter the number corresponding to the action they want to take

    """
    ENV.render()
    for game_index in range(NUM_GAMES):
        done = False
        ENV.reset()
        print(f"========================================Starting game {game_index}========================================")
        while not done:
            action_0 = get_user_input('agent_0')
            action_1 = get_user_input('agent_1')
            actions = {'agent_0': action_0, 'agent_1': action_1}
            obs, reward, done, _, _ = ENV.step(actions)
            print_info(obs, reward, done)
            ENV.render()
        print(f"========================================Game {game_index} over========================================")

if __name__ == '__main__':
    main()