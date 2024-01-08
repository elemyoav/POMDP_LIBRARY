class Translator:

    def __init__(self, num_rocks):
        self.num_rocks = num_rocks

    
    def is_idle_action(self, action):
        return action == 0
    
    def is_move_action(self, action):
        return action >= 1 and action <= 4
    
    def is_sense_action(self, action):
        return action >= 5 and action < 5 + self.num_rocks
    
    def is_sample_action(self, action):
        return action >= 5 + self.num_rocks and action < 5 + 2 * self.num_rocks

    def get_move_direction(self, action):
        if action == 1:
            return 'Up'
        elif action == 2:
            return 'Down'
        elif action == 3:
            return 'Left'
        elif action == 4:
            return 'Right'
        
    def get_sensed_rock_id(self, action):
        return action - 5
    
    def get_sampled_rock_id(self, action):
        return action - 5 - self.num_rocks