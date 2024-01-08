# Tiger Environment

## Introduction
In the Tiger environment, two agents are faced with a decision-making challenge. There are two doors; behind one door lies a pot of gold, and behind the other, a tiger lurks. The agents must decide which door to open in the hope of finding the gold. Their only available action is to listen to the door to discern if the tiger is behind it.

## Configuration
- **Number of Agents**: 2
- **Config**: None
- **Listening Accuracy**: When an agent chooses to listen to a door, there is an 80% probability that the result correctly indicates whether the tiger is behind that door.

## Possible Actions (Per Agent)
1. Open the left door.
2. Open the right door.
3. Listen to the left door.
4. Listen to the right door.

## Rewards (Per Agent)
- **Opening a door with gold**: +10 points.
- **Opening a door with the tiger**: -50 points.
- **Listening action**: -1 point.
> **Note**: All the reward function values can be changed in the `rewards.py` file in this directory

## Observation Space
The observation space is a binary 2-element tuple ({0,1} X {0,1}), where a '1' in any position indicates the observation that the tiger is behind the corresponding door.

