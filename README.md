# DS440-Capstone
Basketball Reinforcement Learning Environment for DS440 Capstone Project

**Note:** Must download basketball_court_half.png image as well as ALL of the UI files, and copy the path of wherever they are saved into the corresponding lines of the code before running

Save the basketball_court_half.png, all the UI files in the UI_files folder in this Github, DS440-frontend-v6.py, and DS440_backend_final_v5.py in the same location when running, otherwise path access will not work

Current Versions of code are in the DS440-frontend-v8.py file for frontend and the DS440_backend_final_v5.py file for backend. The backend can be run on its own with default settings

## Table of Contents

1. [Features](#features)
2. [Prerequisites](#prerequisites)
3. [Code Overview](#code-overview)

## Features

- **Distance-based success probabilities** for shooting (2-pointer or 3-pointer logic).
- **Penalties and rewards** are applied dynamically for different actions.
- **5 players vs 5 defenders constraints** (the 5 players are controlled by the player agent and 5 defenders are an adversarial component to the environment but does not get trained).
- **Move, shoot, pass actions** can be taken by the Player agents
- **Chance for player with ball to "blow by" defenders** Defenders set to algorithmically play man coverage on Players based on assignments, "blowing by" defender allows greater chance to score
- **Shot Made, Shot Missed, Ball Lost, Pass Intercepted** are the 4 terminal states for a given episode/"Step", by default runs for 100 Steps
- **Ball position denoted by yellow 'X' mark in video** player who has ball is highlighted in cyan amongst blue teammates
- **Timelapse video generated** showing results after each step at the end of training, shows shooting proportion, passes completed and intercepted, amount of balls lost, last terminal action, etc

## Prerequisites

Make sure you have Python installed along with the required libraries:
```bash
pip install matplotlib numpy imageio[ffmpeg] cv2 collections torch PyQt6 sqlite3
```

## Code Overview

### `distance(a, b)`
Calculates the Euclidean distance between two points on the court, used to measure distances between players, defenders, and the hoop.

---

### `calculate_shot_success(pos, defender_pos)`
Returns the probability of a successful shot based on the player’s distance to the hoop and the defender’s proximity. Adjusts success rates for 2-pointers and 3-pointers, taking into account defender interference.

---

### `move_toward(positions, idx, target)`
Moves a player toward a target location on the court. Updates the player’s position incrementally by a fixed step size, ensuring the player stays within court boundaries.

---

### `move_defenders()`
Positions each defender to cover their assigned player. Implements collision avoidance to prevent overlapping, simulating realistic defender movement.

---

### `select_action(state)`
Uses an epsilon-greedy strategy to select an action based on the Q-values from the model. The agent either explores with a random action or exploits by choosing the action with the highest Q-value.

---

### `train_model(batch_size=32)`
Trains the DQN model using experience replay, sampling past experiences to improve Q-value estimates and enhance decision-making over time.

---

### `clamp(value, min_value, max_value)`
Constrains a value within specified boundaries. Used to keep player and defender positions within the limits of the court.

---

### `visualize_training(step, result_text)`
Creates a visual representation of each training step. Plots the player, defenders, hoop, and ball, displaying relevant stats for each step. Saves each frame for use in video compilation.

---

### Training Loop
Runs the main training loop, where each step ends in a terminal action: shooting, passing (intercepted or completed), or losing the ball. Updates team stats and calls `visualize_training` at the end of each step.

---

### Video Creation
Compiles the saved frames into a time-lapse video to visualize the training. Each frame represents a single step, allowing a sequential review of player actions and training progress.

--- 
