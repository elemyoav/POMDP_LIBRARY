import gymnasium as gym
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import numpy as np
from collections import OrderedDict
from envs.box_pushing.translator import Translator
from envs.box_pushing.grid import Grid

class DecBoxPushing(MultiAgentEnv):
    def __init__(self, env_config={}):
        num_agents:int = env_config.get('num_agents', 2) # number of agents
        grid_size = env_config.get('grid_size', (2, 2)) # of the form (x, y)
        num_light_boxes:int = env_config.get('num_light_boxes', 1)
        num_heavy_boxes:int = env_config.get('num_heavy_boxes', 1)
        self.p_push:float = env_config.get('p_push', 0.8) # probability of pushing a box in the intended direction
        self.horizon:int = env_config.get('horizon', 300)

        self.grid = Grid(num_agents, num_light_boxes, num_heavy_boxes, grid_size)
        self.translator = Translator(num_agents, num_light_boxes, num_heavy_boxes)

        self.current_step = 0
        self._agent_ids = [f'agent_{i}' for i in range(num_agents)]

        self.action_space = gym.spaces.Discrete(
            1 + # idle
            4 + # move in any direction
            num_light_boxes + # sense b_i
            num_heavy_boxes + # sense B_i
            num_light_boxes*4 + # push b_i in any direction
            num_heavy_boxes*4 # collab push B_i in any direction
        )
        
        self.observation_space = gym.spaces.Dict({
            'location': gym.spaces.MultiDiscrete([*grid_size]),
            'sensed_box': gym.spaces.Discrete(2)  # Boolean (True/False)
        })
        
        super().__init__()

    def reset(self, *, seed=None, options=None):

        self.current_step = 0
        self.grid.reset_board()
        self.translator.reset_box_pushers()

        observations = {
            agent_id: OrderedDict({
                'location': self.grid.get_agent_location(agent_id),
                'sensed_box': 0
            }) for agent_id in self._agent_ids
        }

        return observations, {}

    def push_light_boxes_reward(self, boxes):
        rewards = {}
        for box_id, directions in boxes.items():
            done_state_start = self.grid.is_light_box_done(box_id)
            for direction in directions:
                if np.random.rand() < self.p_push:
                    self.grid.move_light_box(box_id, direction)
            done_state_end = self.grid.is_light_box_done(box_id)

        if done_state_start != done_state_end:
            rewards[box_id] = 500
        else:
            rewards[box_id] = 0
        return rewards
            
    def push_heavy_boxes_reward(self, boxes):
        rewards = {}
        for box_id, directions in boxes.items():
            done_state_start = self.grid.is_heavy_box_done(box_id)
            for direction in directions:
                if np.random.rand() < self.p_push:
                    self.grid.move_heavy_box(box_id, direction)
            done_state_end = self.grid.is_heavy_box_done(box_id)

            if done_state_start != done_state_end:
                rewards[box_id] = 1000
            else:
                rewards[box_id] = 0
        return rewards
    
    def step(self, action_dict):
        self.current_step += 1
        observations, rewards, dones, truncs, infos = {}, {}, {}, {}, {}
        
        for agent_id, action in action_dict.items():
            observations[agent_id] = OrderedDict({
                                        'location': self.grid.get_agent_location(agent_id),
                                        'sensed_box': 0
                                    })
            
            if self.translator.is_idle_action(action):
                rewards[agent_id] = 0

            if self.translator.is_move_agent_action(action):
                direction = self.translator.get_move_agent_direction(action)
                self.grid.move_agent(agent_id, direction)
                observations[agent_id]['location'] = self.grid.get_agent_location(agent_id)
                rewards[agent_id] = -10
            
            if self.translator.is_sense_light_box_action(action):
                box_num = self.translator.get_sense_light_box_num(action)
                box_id = f'light_box_{box_num}'
                observations[agent_id]['sensed_box'] = self.grid.sense_light_box(agent_id, box_id)
                rewards[agent_id] = -1
            
            if self.translator.is_sense_heavy_box_action(action):
                box_num = self.translator.get_sense_heavy_box_num(action)
                box_id = f'heavy_box_{box_num}'
                observations[agent_id]['sensed_box'] = self.grid.sense_heavy_box(agent_id, box_id)
                rewards[agent_id] = -1
            
            if self.translator.is_push_light_box_action(action):
                rewards[agent_id] = -30
                box_num = self.translator.get_push_light_box_num(action)
                box_id = f'light_box_{box_num}'
                direction = self.translator.get_push_light_box_direction(action)

                if self.grid.can_push_light_box(agent_id, box_id):
                    self.translator.add_light_box_pusher(box_id, direction, agent_id)
            
            if self.translator.is_push_heavy_box_action(action):
                rewards[agent_id] = -20
                box_num = self.translator.get_push_heavy_box_num(action)
                box_id = f'heavy_box_{box_num}'
                direction = self.translator.get_push_heavy_box_direction(action)

                if self.grid.can_push_heavy_box(agent_id, box_id):
                    self.translator.add_heavy_box_pusher(box_id, direction, agent_id)
            
        pushing_results = self.translator.get_box_pushing_directions()

        light_boxes_rewards = self.push_light_boxes_reward(pushing_results['light_boxes_directions'])
        heavy_boxes_rewards = self.push_heavy_boxes_reward(pushing_results['heavy_boxes_directions'])

        light_boxes_succ_agents = pushing_results['light_boxes_succ_pushers']
        heavy_boxes_succ_agents = pushing_results['heavy_boxes_succ_pushers']

        for box_id, agents in light_boxes_succ_agents.items():
            for agent_id in agents:
                rewards[agent_id] += light_boxes_rewards[box_id]

        for box_id, agents in heavy_boxes_succ_agents.items():
            for agent_id in agents:
                rewards[agent_id] += heavy_boxes_rewards[box_id]

        dones['__all__'] = self.grid.is_game_over() or self.current_step >= self.horizon
        truncs['__all__'] = False

        return observations, rewards, dones, truncs, infos


                