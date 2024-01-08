import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import random
import numpy as np

from envs.tiger.rewards import OPEN_TIGER_REWARD, OPEN_MONEY_REWARD, LISTEN_REWARD


# Observation space: {0, 1} x {0, 1}
NULL_OBS = np.array([0, 0])
LEFT_OBS = np.array([1, 0])
RIGHT_OBS = np.array([0, 1])
NOISE_OBS = np.array([1, 1])

# Action space: {0, 1, 2, 3}
OPEN_LEFT = 0
OPEN_RIGHT = 1
LISTEN_LEFT = 2
LISTEN_RIGHT = 3

#render options
AGENT_0 = '🕵🏼'
AGENT_1 = '🕵🏻'
LEFT_DOOR = '🚪'
RIGHT_DOOR = '🚪'

class DecTigerEnv(MultiAgentEnv):
    def __init__(self, config={}):

        # Open left, open right, listen_left, listen_right
        self.observation_space = gym.spaces.MultiDiscrete(
            [2, 2])  # {0, 1} x {0, 1}

        # Open left, open right, listen_left, listen_right
        self.action_space = gym.spaces.Discrete(4)  # {0, 1, 2 ,3}

        # Initialize state and other variables
        self.state = NULL_OBS
        self._agent_ids = ['agent_0', 'agent_1']

        super().__init__()

    def reset(self, *, seed=None, options=None,):
        # Reset the state (randomly place the tiger behind one of the two doors)
        self._set_render_options(agent_0='🕵🏼', agent_1='🕵🏻', left_door='🚪', right_door='🚪')

        self.state = random.choice([LEFT_OBS, RIGHT_OBS])

        # Observations for each agent (initially, both agents have the same observation)
        observations = {agent_id: NULL_OBS for agent_id in self._agent_ids}
        return observations, {}

    def step(self, action_dict):

        observations, rewards, dones, truncs, infos = {}, {}, {}, {}, {}

        truncs['__all__'] = False

        if OPEN_LEFT in action_dict.values() or OPEN_RIGHT in action_dict.values():
            # if either agent opens a door, the episode is over
            dones['__all__'] = True
        else:
            dones['__all__'] = False

        for agent_id in action_dict.keys():
            infos[agent_id] = {}  # set info to be constant
            # get the reward for the agent's action
            rewards[agent_id] = self._agent_reward(agent_id, action_dict[agent_id])
            # get the observation for the agent's action
            observations[agent_id] = self._agent_obs(agent_id, action_dict[agent_id])

        return observations, rewards, dones, truncs, infos

    def render(self, mode='human'):
        print(f"""
        {LEFT_DOOR} {RIGHT_DOOR}
        {AGENT_0} {AGENT_1}
        """
        )

    def _agent_reward(self, agent_id, action):
        if action == OPEN_LEFT:
            if self._is_tiger_left():
                self._set_render_options(**{
                    'left_door': '🐅',
                     agent_id: '😱'
                })
                return OPEN_TIGER_REWARD
            else:
                self._set_render_options(**{
                    'left_door': '💰',
                    agent_id: '🤑'
                })
                return OPEN_MONEY_REWARD

        if action == OPEN_RIGHT:
            if self._is_tiger_right():
                self._set_render_options(**{
                    'right_door': '🐅',
                    agent_id: '😱'
                })
                return OPEN_TIGER_REWARD
            else:
                self._set_render_options(**{
                    'right_door': '💰',
                    agent_id: '🤑'
                })
                return OPEN_MONEY_REWARD

        if action == LISTEN_LEFT:
            self._set_render_options(**{
                agent_id: '👈👂🏼'
            })
            return LISTEN_REWARD

        if action == LISTEN_RIGHT:
            self._set_render_options(**{
                agent_id: '👉👂🏻'
            })
            return LISTEN_REWARD

    def _is_tiger_left(self):
        return np.array_equal(self.state, LEFT_OBS)

    def _is_tiger_right(self):
        return np.array_equal(self.state, RIGHT_OBS)

    def _agent_obs(self, agent_id, action):
        if action == OPEN_LEFT or action == OPEN_RIGHT:
            return self.state
        return self._listen_obs(action)

    def _listen_obs(self, action):
        # this function assumes the action is either listen left or listen right
        p = 0
        if (action == LISTEN_LEFT and self._is_tiger_left()) or (action == LISTEN_RIGHT and self._is_tiger_right()):
            p = 0.75

        if random.random() < p:
            return self.state
        return NULL_OBS

    def _set_render_options(self, agent_0=None, agent_1=None, left_door=None, right_door=None):
        global AGENT_0, AGENT_1, LEFT_DOOR, RIGHT_DOOR
        AGENT_0 = agent_0 if agent_0 else AGENT_0
        AGENT_1 = agent_1 if agent_1 else AGENT_1
        LEFT_DOOR = left_door if left_door else LEFT_DOOR
        RIGHT_DOOR = right_door if right_door else RIGHT_DOOR