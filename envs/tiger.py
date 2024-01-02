import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import random
from typing import Callable, Dict, List, Tuple, Optional, Union, Set, Type
from ray.rllib.utils.typing import (
    AgentID,
    EnvCreator,
    EnvID,
    EnvType,
    MultiAgentDict,
    MultiEnvDict,
)
from ray.rllib.utils.annotations import PublicAPI


class DecTigerEnv(MultiAgentEnv):
    def __init__(self, config={}):
        # Define the observation and action spaces
        # Each element in the tuple can be 0 or 1 (2 options)
        element_space = gym.spaces.Discrete(2)

        # Define the observation space as a tuple of two elements
        self.observation_space = gym.spaces.Tuple(
            (element_space, element_space))# M
        # Open left, open right, listen_left, listen_right
        self.action_space = gym.spaces.Discrete(4)# M
        
        # Initialize state and other variables
        self.state = (0, 0)
        self.num_agents = 2
        self._agent_ids = set(['agent_0', 'agent_1']) # M

        super().__init__()

    @PublicAPI
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ) -> Tuple[MultiAgentDict, MultiAgentDict]:
        # Reset the state (randomly place the tiger behind one of the two doors)
        self.state = random.choice([(0, 1), (1, 0)])
        self.done = False
        # Observations for each agent (initially, both agents have the same observation)
        # TODO: make them have random observations at first
        observations = { agent_id: (0,0) for agent_id in self._agent_ids }
        return observations, {}

    @PublicAPI
    def step(
        self, action_dict: MultiAgentDict
    ) -> Tuple[
        MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict, MultiAgentDict
    ]:
        rewards, terminateds, truncateds, infos = {}, {}, {}, {}

        # set info to be constant
        infos["agent_0"] = {}
        infos["agent_1"] = {}

        # set truncated to be constant
        truncateds['__all__'] = False

        # terminated only if one of the agents picked open left or open right
        if action_dict["agent_0"] == 0 or action_dict["agent_0"] == 1 or action_dict["agent_1"] == 0 or action_dict["agent_1"] == 1:
            terminateds['__all__'] = True
            observations = {
                agent_id: self.state for agent_id in self._agent_ids}
        else:
            terminateds['__all__'] = False

        if action_dict["agent_0"] == 0 and action_dict["agent_1"] == 0:
            # Both agents open left door
            if self.state == (1, 0):
                # Tiger is behind left door
                rewards["agent_0"] = -50
                rewards["agent_1"] = -50
            else:
                # Tiger is behind right door
                rewards["agent_0"] = 10
                rewards["agent_1"] = 10

        elif action_dict["agent_0"] == 0 and action_dict["agent_1"] == 1:
            # agent 0 opens left door, agent 1 opens right door
            rewards["agent_0"] = -100
            rewards["agent_1"] = -100

        elif action_dict["agent_0"] == 0 and action_dict["agent_1"] == 2:
            # agent 0 opens left door, agent 1 listens left
            if self.state == (1, 0):
                # Tiger is behind left door
                rewards["agent_0"] = -50
                rewards["agent_1"] = -1
            else:
                # Tiger is behind right door
                rewards["agent_0"] = 10
                rewards["agent_1"] = -1

        elif action_dict["agent_0"] == 0 and action_dict["agent_1"] == 3:
            # agent 0 opens left door, agent 1 listens right
            if self.state == (1, 0):
                # Tiger is behind left door
                rewards["agent_0"] = -50
                rewards["agent_1"] = -1
            else:
                # Tiger is behind right door
                rewards["agent_0"] = 10
                rewards["agent_1"] = -1

        elif action_dict["agent_0"] == 1 and action_dict["agent_1"] == 0:
            # agent 0 opens right door, agent 1 opens left door
            rewards["agent_0"] = -100
            rewards["agent_1"] = -100

        elif action_dict["agent_0"] == 1 and action_dict["agent_1"] == 1:
            # Both agents open right door
            if self.state == (1, 0):
                # Tiger is behind left door
                rewards["agent_0"] = 10
                rewards["agent_1"] = 10
            else:
                # Tiger is behind right door
                rewards["agent_0"] = -50
                rewards["agent_1"] = -50
        elif action_dict["agent_0"] == 1 and action_dict["agent_1"] == 2:
            # agent 0 opens right door, agent 1 listens left
            if self.state == (1, 0):
                # Tiger is behind left door
                rewards["agent_0"] = 10
                rewards["agent_1"] = -1
            else:
                # Tiger is behind right door
                rewards["agent_0"] = -50
                rewards["agent_1"] = -1

        elif action_dict["agent_0"] == 1 and action_dict["agent_1"] == 3:
            # agent 0 opens right door, agent 1 listens right
            if self.state == (1, 0):
                # Tiger is behind left door
                rewards["agent_0"] = 10
                rewards["agent_1"] = -1
            else:
                # Tiger is behind right door
                rewards["agent_0"] = -50
                rewards["agent_1"] = -1

        elif action_dict["agent_0"] == 2 and action_dict["agent_1"] == 0:
            # agent 0 listens left, agent 1 opens left door
            if self.state == (1, 0):
                # Tiger is behind left door
                rewards["agent_0"] = -1
                rewards["agent_1"] = -50
            else:
                # Tiger is behind right door
                rewards["agent_0"] = -1
                rewards["agent_1"] = 10

        elif action_dict["agent_0"] == 2 and action_dict["agent_1"] == 1:
            # agent 0 listens left, agent 1 opens right door
            if self.state == (1, 0):
                # Tiger is behind left door
                rewards["agent_0"] = -1
                rewards["agent_1"] = 10
            else:
                # Tiger is behind right door
                rewards["agent_0"] = -1
                rewards["agent_1"] = -50

        elif action_dict["agent_0"] == 2 and action_dict["agent_1"] == 2:
            # agent 0 listens left, agent 1 listens left
            rewards["agent_0"] = -1
            rewards["agent_1"] = -1

            if self.state == (1, 0):
                # Tiger is behind left door
                o = self.get_obs(0.25)
                observations = {agent_id: o for agent_id in self._agent_ids}
            else:
                # Tiger is behind right door
                observations = {agent_id: (0, 0)
                                for agent_id in self._agent_ids}

        elif action_dict["agent_0"] == 2 and action_dict["agent_1"] == 3:
            # agent 0 listens left, agent 1 listens right
            rewards["agent_0"] = -1
            rewards["agent_1"] = -1

            if self.state == (1, 0):
                # Tiger is behind left door
                observations = {"agent_0": self.get_obs(
                    0.125), "agent_1": (0, 0)}
            else:
                # Tiger is behind right door
                observations = {"agent_0": (
                    0, 0), "agent_1": self.get_obs(0.125)}

        elif action_dict["agent_0"] == 3 and action_dict["agent_1"] == 0:
            # agent 0 listens right, agent 1 opens left door
            if self.state == (1, 0):
                # Tiger is behind left door
                rewards["agent_0"] = -1
                rewards["agent_1"] = -50
            else:
                # Tiger is behind right door
                rewards["agent_0"] = -1
                rewards["agent_1"] = 10

        elif action_dict["agent_0"] == 3 and action_dict["agent_1"] == 1:
            # agent 0 listens right, agent 1 opens right door
            if self.state == (1, 0):
                # Tiger is behind left door
                rewards["agent_0"] = -1
                rewards["agent_1"] = 10
            else:
                # Tiger is behind right door
                rewards["agent_0"] = -1
                rewards["agent_1"] = -50

        elif action_dict["agent_0"] == 3 and action_dict["agent_1"] == 2:
            # agent 0 listens right, agent 1 listens left
            rewards["agent_0"] = -1
            rewards["agent_1"] = -1

            if self.state == (1, 0):
                # Tiger is behind left door
                observations = {"agent_0": (
                    0, 0), "agent_1": self.get_obs(0.125)}
            else:
                # Tiger is behind right door
                observations = {"agent_0": self.get_obs(
                    0.125), "agent_1": (0, 0)}

        elif action_dict["agent_0"] == 3 and action_dict["agent_1"] == 3:
            # agent 0 listens right, agent 1 listens right
            rewards["agent_0"] = -1
            rewards["agent_1"] = -1

            if self.state == (1, 0):
                # Tiger is behind left door
                observations = {agent_id: (0, 0)
                                for agent_id in self._agent_ids}
            else:
                # Tiger is behind right door
                o = self.get_obs(0.25)
                observations = {agent_id: o for agent_id in self._agent_ids}

        return observations, rewards, terminateds, truncateds, infos

    # Necessary for compatibility with Ray RLlib
    def observation_space_sample(self, agent_ids=None):
        return {agent_id: self.observation_space.sample() for agent_id in self._agent_ids}

    def action_space_sample(self, agent_ids=None):
        return {agent_id: self.action_space.sample() for agent_id in self._agent_ids}

    def get_agent_ids(self):
        return self._agent_ids

    def get_obs(self, p):
        # this is used to give an agent an observation, with probability p
        # it will be a guess, otherwise it will be the true state
        if random.random() < p:
            return (0, 0)
        else:
            return self.state
