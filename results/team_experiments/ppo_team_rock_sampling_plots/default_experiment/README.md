### Domain Summary: Two-Rover Grid Experiment

#### Experiment Setup:
- *Grid Dimensions:* 3 (width) x 5 (height)
- *Divided Areas:*
  - *Rover 1 Area:* Columns 0-1
  - *Shared Area:* Columns 2
  - *Rover 2 Area:* Columns 3-4
- *Rocks Distribution:* Each area contains 1 rocks.
- *Rock Quality Assessment:* The probability of a successful rock sample is determined by the formula:
    P(successful sample) = exp(-|rock_location - rover_location|)
    This formula indicates that the closer the rover is to the rock, the higher the probability of a successful sample.

#### Learning Parameters:
- *Learning Rates Sampled:* 0.0001
- *Discount Factor Sampled:* 0.9, 0.99

#### Experiment Details:
- *Problem:* Team-based box pushing.
- *Algorithm:* Proximal Policy Optimization (PPO).