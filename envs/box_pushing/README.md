# Box Pushing Environment

## Introduction
In the Box Pushing environment, agents are tasked with moving boxes to a target position within a grid. There are two types of boxes: light and heavy. A single agent can push a light box, but pushing a heavy box requires the collaboration of two agents.

## Configuration
- **Number of Agents**: `num_agents` (integer)
- **Grid Size**: `grid_size` (tuple of integers, width and height)
- **Number of Light Boxes**: `num_light_boxes`
- **Number of Heavy Boxes**: `num_heavy_boxes`
- **Push Probability**: `p_push` (probability that a push action will succeed)
- **Horizon**: `horizon` (integer)

## Action Space (Per Agent)
Agents can perform the following actions:
1. Idle
2. Move in any direction (up, down, left, right)
3. Sense box
4. Push box

## Rewards
- **Idle Action**: 0 points
- **Move Action**: -10 points
- **Sense Action**: -1 point
- **Push Light Box (Non-Collaborative)**: -30 points
- **Push Heavy Box (Collaborative)**: -20 points
- **Getting a Light Box to the Target**: +500 points (awarded to all contributing agents)
- **Getting a Heavy Box to the Target**: +1000 points (awarded to all contributing agents)
> **Note**: All the reward function values can be changed in the `reward.py` file in this directory

## Observation Space
The observation space for each agent includes:
- The location of the agent (grid coordinates).
- A boolean indicating whether the sense action succeeded.

