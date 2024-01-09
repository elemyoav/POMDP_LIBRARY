import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--env', type=str, default='dec_tiger')
parser.add_argument('--num_games', type=int, default=3)

ARGS = parser.parse_args()