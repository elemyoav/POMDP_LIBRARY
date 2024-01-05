from gym.spaces import Discrete, MultiDiscrete, Dict
from ray.rllib import MultiAgentEnv
import numpy as np
from collections import OrderedDict
from numpy import array, array_equal

class DecBoxPushing(MultiAgentEnv):
    def __init__(self, env_config:dict):
        self.num_agents:int = env_config.get('num_agents', 2) # number of agents
        self.grid_size = env_config.get('grid_size', (2, 2)) # of the form (x, y)
        self.num_light_boxes:int = env_config.get('num_light_boxes', 1)
        self.num_heavy_boxes:int = env_config.get('num_heavy_boxes', 1)
        self.p_push:float = env_config.get('p_push', 0.8) # probability of pushing a box in the intended direction

        self.grid:list = [ array([x, y]) for x in range(self.grid_size[0]) for y in range(self.grid_size[1]) ]

        self.state = {
            **{f'agent_{i}': array([0, 0]) for i in range(self.num_agents)},
            **{f'light_box_{i}': {'location': array([0, 0]), 'done': False} for i in range(self.num_light_boxes)},
            **{f'heavy_box_{i}': {'location': array([0, 0]), 'done': False} for i in range(self.num_heavy_boxes)},
            **{'target_location': array([0, 0])}
        }

        self._agent_ids = [f'agent_{i}' for i in range(self.num_agents)]

        self.action_space = Discrete(
            1 + # idle
            4 + # move in any direction
            self.num_light_boxes + # sense b_i
            self.num_heavy_boxes + # sense B_i
            self.num_light_boxes*4 + # push b_i in any direction
            self.num_heavy_boxes*4 # collab push B_i in any direction
        )
        
        self.observation_space = Dict({
            'location': MultiDiscrete([*self.grid_size]),
            'sensed_box': Discrete(2)  # Boolean (True/False)
        })

        super().__init__()

    def _is_idle_action(self, action):
        return action == 0
    
    def _is_move_action(self, action):
        return action >= 1 and action <= 4
    
    def _is_sense_small_box_action(self, action):
        return action >= 5 and action < 5 + self.num_light_boxes

    def _is_sense_large_box_action(self, action):
        return action >= 5 + self.num_light_boxes and action < 5 + self.num_light_boxes + self.num_heavy_boxes

    def _is_push_small_box_action(self, action):
        return action >= 5 + self.num_light_boxes + self.num_heavy_boxes and action < 5 + self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4

    def _is_push_large_box_action(self, action):
        return action >= 5 + self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 and action < 5 + self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + self.num_heavy_boxes*4


    def _get_random_loc(self):
        return self.grid[np.random.randint(len(self.grid))]
    
    def _get_random_loc_not_on_boxes(self):
        loc = self._get_random_loc()
        for i in range(self.num_light_boxes):
            if array_equal(loc, self.state[f'light_box_{i}']['location']):
                return self._get_random_loc_not_on_boxes()
        for i in range(self.num_heavy_boxes):
            if array_equal(loc, self.state[f'heavy_box_{i}']['location']):
                return self._get_random_loc_not_on_boxes()
        return loc

    def reset(self, *, seed=None, options=None):

        for agent_id in self._agent_ids:
            self.state[agent_id] = self._get_random_loc()
        
        for i in range(self.num_light_boxes):
            self.state[f'light_box_{i}']['location'] = self._get_random_loc()
            self.state[f'light_box_{i}']['done'] = False
        
        for i in range(self.num_heavy_boxes):
            self.state[f'heavy_box_{i}']['location'] = self._get_random_loc()
            self.state[f'heavy_box_{i}']['done'] = False
        
        self.state['target_location'] = self._get_random_loc_not_on_boxes()

        observations = {
            agent_id: OrderedDict({
                'location': self.state[agent_id],
                'sensed_box': 0
            }) for agent_id in self._agent_ids
        }

        return observations, {}


    def _move_agent(self, action, agent_id):
        directions = {
            1: 'Left',
            2: 'Right',
            3: 'Up',
            4: 'Down'
        }
        
        direction = directions[action]
        agent_x, agent_y = self.state[agent_id]

        if direction == 'Left':
            agent_x -= 1
        elif direction == 'Right':
            agent_x += 1
        elif direction == 'Up':
            agent_y += 1
        elif direction == 'Down':
            agent_y -= 1

        if agent_x < 0 or agent_x >= self.grid_size[0]:
            agent_x = self.state[agent_id][0]

        if agent_y < 0 or agent_y >= self.grid_size[1]:
            agent_y = self.state[agent_id][1]
        
        self.state[agent_id] = (agent_x, agent_y)
    
    def _sense_small_box(self, action, agent_id):
        box_id = action - 5
        return int(array_equal(self.state[f'light_box_{box_id}']['location'], self.state[agent_id]))
    
    def _sense_large_box(self, action, agent_id):
        box_id = action - 5 - self.num_light_boxes
        return int(array_equal(self.state[f'heavy_box_{box_id}']['location'], self.state[agent_id]))
    
    def _move_small_box(self, box_id, push_action):
        
        directions = {
            5 +self.num_light_boxes + self.num_heavy_boxes + box_id*4 + 0: 'Left',
            5 +self.num_light_boxes + self.num_heavy_boxes + box_id*4 + 1: 'Right',
            5 +self.num_light_boxes + self.num_heavy_boxes + box_id*4 + 2: 'Up',
            5 +self.num_light_boxes + self.num_heavy_boxes + box_id*4 + 3: 'Down'
        }

        direction = directions[push_action]
        box_x, box_y = self.state[f"light_box_{box_id}"]['location']

        if direction == 'Left':
            box_x -= 1
        elif direction == 'Right':
            box_x += 1
        elif direction == 'Up':
            box_y += 1
        elif direction == 'Down':
            box_y -= 1
        
        if box_x < 0 or box_x >= self.grid_size[0]:
            box_x = self.state[f"light_box_{box_id}"]['location'][0]
        
        if box_y < 0 or box_y >= self.grid_size[1]:
            box_y = self.state[f"light_box_{box_id}"]['location'][1]

        self.state[f"light_box_{box_id}"]['location'] = (box_x, box_y)

        if array_equal(self.state[f"light_box_{box_id}"]['location'], self.state['target_location']):
            self.state[f"light_box_{box_id}"]['done'] = True
    
    def _move_big_box(self, box_id, push_action):
        
        directions = {
            5 +self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + box_id*4 + 0: 'Left',
            5 +self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + box_id*4 + 1: 'Right',
            5 +self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + box_id*4 + 2: 'Up',
            5 +self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + box_id*4 + 3: 'Down'
        }

        direction = directions[push_action]
        box_x, box_y = self.state[f"heavy_box_{box_id}"]['location']

        if direction == 'Left':
            box_x -= 1
        elif direction == 'Right':
            box_x += 1
        elif direction == 'Up':
            box_y += 1
        elif direction == 'Down':
            box_y -= 1
        
        if box_x < 0 or box_x >= self.grid_size[0]:
            box_x = self.state[f"heavy_box_{box_id}"]['location'][0]
        
        if box_y < 0 or box_y >= self.grid_size[1]:
            box_y = self.state[f"heavy_box_{box_id}"]['location'][1]

        self.state[f"heavy_box_{box_id}"]['location'] = (box_x, box_y)

        if array_equal(self.state[f"heavy_box_{box_id}"]['location'], self.state['target_location']):
            self.state[f"heavy_box_{box_id}"]['done'] = True
    
    def _can_push_small_box(self, box_id, agent_id):
        return array_equal(self.state[f'light_box_{box_id}']['location'], self.state[agent_id]) and not self.state[f'light_box_{box_id}']['done']
    
    def _can_push_large_box(self, box_id, agent_id):
        return array_equal(self.state[f'heavy_box_{box_id}']['location'], self.state[agent_id]) and not self.state[f'heavy_box_{box_id}']['done']
    
    def _is_done(self):
        # done when all boxes are at the target location
        return all([self.state[f'light_box_{i}']['done'] for i in range(self.num_light_boxes)]) and all([self.state[f'heavy_box_{i}']['done'] for i in range(self.num_heavy_boxes)])
    
    def step(self, action_dict):
        observations, rewards, dones, truncs, infos = {}, {}, {}, {}, {}

        dones['__all__'] = self._is_done()
        truncs['__all__'] = False
        
        small_box_pushers = {
            f'light_box_{i}': [] for i in range(self.num_light_boxes)
        }
        large_box_pushers = {
            f'heavy_box_{i}': {
                'Left': [],
                'Right': [],
                'Up': [],
                'Down': []
            } 
            for i in range(self.num_heavy_boxes)
        }

        for agent_id, action in action_dict.items():
            observations[agent_id] = OrderedDict({
                                        'location': self.state[agent_id],
                                        'sensed_box': 0
                                    })
            
            if self._is_idle_action(action):
                rewards[agent_id] = 0
            
            if self._is_move_action(action):
                self._move_agent(action, agent_id)
                rewards[agent_id] = -10
                observations[agent_id]['location'] = self.state[agent_id]
            
            if self._is_sense_small_box_action(action):
                observations[agent_id]['sensed_box'] = self._sense_small_box(action, agent_id)
                rewards[agent_id] = -1
            
            if self._is_sense_large_box_action(action):
                observations[agent_id]['sensed_box'] = self._sense_large_box(action, agent_id)
                rewards[agent_id] = -1
            
            if self._is_push_small_box_action(action):
                rewards[agent_id] = -30
                box_id = action - 5 - self.num_light_boxes - self.num_heavy_boxes

                # if the box can be pushed
                if self._can_push_small_box(box_id, agent_id):
                    # if no one pushed the box yet
                    if small_box_pushers[f'light_box_{box_id}'] == []:
                        small_box_pushers[f'light_box_{box_id}'] = [agent_id]
                        self._move_small_box(box_id, action)
                        if array_equal(self.state[f'light_box_{box_id}']['loc'], self.state['target_location']):
                            rewards[agent_id] = 500
                
            if self._is_push_large_box_action(action):
                rewards[agent_id] = -20
                box_id = action - 5 - self.num_light_boxes - self.num_heavy_boxes - self.num_light_boxes*4

                # if the box can be pushed
                if self._can_push_large_box(box_id, agent_id):
                    directions = {
                        5 + self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + box_id*4 + 0: 'Left',
                        5 + self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + box_id*4 + 1: 'Right',
                        5 + self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + box_id*4 + 2: 'Up',
                        5 + self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + box_id*4 + 3: 'Down'
                    }
                    direction = directions[action]
                    large_box_pushers[f'heavy_box_{box_id}'][direction].append(agent_id)

            
        for large_box_id, pushers in large_box_pushers.items():
            
            left_pushers = pushers['Left']
            right_pushers = pushers['Right']
            up_pushers = pushers['Up']
            down_pushers = pushers['Down']

            push_left = len(left_pushers) - len(right_pushers)

            if push_left >= 2:
                # move the box left
                box_done_state = self.state[f'heavy_box_{large_box_id}']['done']
                self._move_big_box(large_box_id, 5 + self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + large_box_id*4 + 0)
                if box_done_state != self.state[f'heavy_box_{large_box_id}']['done']:
                    for pusher in left_pushers:
                        rewards[pusher] += 1000

            if push_left <= -2:
                # move the box right
                box_done_state = self.state[f'heavy_box_{large_box_id}']['done']
                self._move_big_box(large_box_id, 5 + self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + large_box_id*4 + 1)

                if box_done_state != self.state[f'heavy_box_{large_box_id}']['done']:
                    for pusher in right_pushers:
                        rewards[pusher] += 1000

            push_up = len(up_pushers) - len(down_pushers)

            if push_up >= 2:
                # move the box up
                box_done_state = self.state[f'heavy_box_{large_box_id}']['done']
                self._move_big_box(large_box_id, 5 + self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + large_box_id*4 + 2)
                if box_done_state != self.state[f'heavy_box_{large_box_id}']['done']:
                    for pusher in up_pushers:
                        rewards[pusher] += 1000

            if push_up <= -2:
                # move the box down
                box_done_state = self.state[f'heavy_box_{large_box_id}']['done']
                self._move_big_box(large_box_id, 5 + self.num_light_boxes + self.num_heavy_boxes + self.num_light_boxes*4 + large_box_id*4 + 3)
                if box_done_state != self.state[f'heavy_box_{large_box_id}']['done']:
                    for pusher in down_pushers:
                        rewards[pusher] += 1000
                
        return observations, rewards, dones, truncs, infos