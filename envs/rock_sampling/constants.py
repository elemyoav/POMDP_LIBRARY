import numpy as np

def default_observation_quality_function(pos1, pos2):
    """
    summary: Default observation quality function. Gives a more accurate observation if the rover is closer to the rock.
    input: pos1, pos2: tuple of (x, y) coordinates
    output: float between 0 and 1
    """
    
    pos1 = np.array(pos1)
    pos2 = np.array(pos2)
    return np.exp(-np.linalg.norm(pos1 - pos2))

DEFAULT_CONFIG = {
        'grid_config': {
            'width': 3,
            'rover1_height': 2,
            'shared_height': 1,
            'rover2_height': 2,
            'num_rocks_rover_1': 1,
            'num_rocks_rover_2': 1,
            'num_rocks_shared': 1,
            'observation_quality_function': default_observation_quality_function
        },
        'horizon': 100
    }