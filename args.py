import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--algo', type=str, default='ppo')
parser.add_argument('--env', type=str, default='box_pushing')
parser.add_argument('--training_iterations', type=int, default=2000)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--lr', type=float, default=5e-5)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--epsilon', type=float, default=1.0)
parser.add_argument('--epsilon_decay', type=float, default=0.9995)
parser.add_argument('--epsilon_min', type=float, default=0.01)
parser.add_argument('--framework', type=str, default='torch')
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--log_level', type=str, default='ERROR')

ARGS = parser.parse_args()