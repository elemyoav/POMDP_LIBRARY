### Domain Summary: Two-Rover Grid Experiment

#### Experiment Setup:
- *Grid Dimensions:* 5 (width) x 8 (height)
- *Divided Areas:*
  - *Rover 1 Area:* Columns 0-2
  - *Shared Area:* Columns 3-4
  - *Rover 2 Area:* Columns 5-7
- *Rocks Distribution:* Each area contains 3 rocks, including one of bad quality.
- *Rock Quality Assessment:* The probability of a successful rock sample is determined by the formula:
    P(successful sample) = exp(-|rock_location - rover_location|)
    This formula indicates that the closer the rover is to the rock, the higher the probability of a successful sample.