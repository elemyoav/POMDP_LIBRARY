import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from envs.rock_sampling.grid import Grid, NULL_QUALITY, BAD_QUALITY, GOOD_QUALITY
from envs.rock_sampling.translator import Translator
from collections import OrderedDict


class DecRockSampling(MultiAgentEnv):

    def __init__(self, env_config):
        self.grid = Grid(env_config['grid_config'])

        self.translator = Translator(self.grid.get_num_rocks())
        self._agent_ids = ['rover1', 'rover2']

        self._observation_space = gym.spaces.Dict({
            'position': gym.spaces.MultiDiscrete([self.grid.area_width, self.grid.area_height]),
            'rock_quality': gym.spaces.MultiDiscrete([3])
        })

        self.action_space = gym.spaces.Discrete(1 + # Idle action
                                                4 + # Move actions
                                                self.grid.get_num_rocks() + # Sense actions
                                                self.grid.get_num_rocks()   # Sample actions
                                                )
    
    def reset(self, *, seed=None, options=None):
        self.grid.reset_board()

        observations = {
            agent_id: OrderedDict({
                'position': self.grid.get_rover_position(agent_id),
                'rock_quality': NULL_QUALITY
            }) for agent_id in self._agent_ids
        }

        return observations, {}
    
    def step(self, action_dict):
        observations, rewards, dones, truncs, infos = {}, {}, {}, {}, {}

        for agent_id, action in action_dict.items():

            observations[agent_id] = OrderedDict({
                'position': self.grid.get_rover_position(agent_id),
                'rock_quality': NULL_QUALITY
            })

            if self.translator.is_idle_action(action):
                rewards[agent_id] = 0
            
            if self.translator.is_move_action(action):
                direction = self.translator.get_move_direction(action)
                self.grid.move_rover(agent_id, direction)
                observations[agent_id]['position'] = self.grid.get_rover_position(agent_id)
                rewards[agent_id] = -1
            
            if self.translator.is_sense_action(action):
                rock_id = self.translator.get_sensed_rock_id(action)
                rock_quality = self.grid.get_rock_quality(rock_id)
                observations[agent_id]['rock_quality'] = rock_quality
                rewards[agent_id] = -5
            
            if self.translator.is_sample_action(action):
                rover_1_area_clear_before = self.grid.is_rover_1_area_clear()
                rover_2_area_clear_before = self.grid.is_rover_2_area_clear()
                shared_area_clear_before = self.grid.is_shared_area_clear()

                rock_id = self.translator.get_sampled_rock_id(action)
                rock_quality = self.grid.sample_rock(agent_id, rock_id)

                if rock_quality is None or rock_quality == 'Bad':
                    rewards[agent_id] = -500
                    continue

                rover_1_area_clear_after = self.grid.is_rover_1_area_clear()
                rover_2_area_clear_after = self.grid.is_rover_2_area_clear()
                shared_area_clear_after = self.grid.is_shared_area_clear()

                if rover_1_area_clear_before != rover_1_area_clear_after:
                    rewards['rover1'] = 750
                    continue
                if rover_2_area_clear_before != rover_2_area_clear_after:
                    rewards['rover2'] = 750
                    continue
                
                if shared_area_clear_before != shared_area_clear_after:
                    rewards['rover1'] = 750
                    rewards['rover2'] = 750
                    continue
            
        dones['__all__'] = self.grid.is_game_over()
        truncs['__all__'] = False
        return observations, rewards, dones, truncs, infos

