# DS440-Capstone
Basketball Reinforcement Learning Environment for DS440 Capstone Project

**Note:** Must download basketball_court_half.png image, and copy the path of wherever it is saved into the code before running

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
5. [Code Overview](#code-overview)

## Features

- **Distance-based success probabilities** for shooting (2-pointer or 3-pointer logic).
- **Penalties and rewards** are applied dynamically for different actions.
- **Energy constraints** affect valid actions (e.g., driving to the basket requires sufficient energy).

## Prerequisites

Make sure you have Python installed along with the required libraries:
```bash
pip install matplotlib numpy
```
## Code Overview

### `distance_from_hoop(x, y)`  
Calculates the Euclidian distance from the player’s position to the hoop located at `(0, 5.2)`.

---

### `calculate_success_prob(dist)`  
Returns the probability of a successful shot based on the distance from the hoop:  
- **2-pointers:** 54.5% success within 23.75 feet, based on the NBA players' average two-pointer percentage.  
- **3-pointers:** 35.3% success beyond 23.75 feet, based on the NBA players' average three-pointer percentage.  
- **Reductions:** Further reductions are applied beyond certain distances to simulate shot difficulty.

---

### `reward(action, state)`  
Calculates the reward or penalty for player actions:  
- **Shoot:** Returns a reward based on shot success or applies a penalty for missed shots.  
- **Pass:** A safe action that provides a moderate reward.  
- **Drive:** Available only with sufficient energy, with probabilistic outcomes based on game conditions.

---

### `display_player_and_hoop(state, step)`  
Visualizes the player and hoop positions on a basketball court using `matplotlib`.  
This function plots the player's position on the court along with the hoop's location and adjusts the court view to match NBA dimensions.

