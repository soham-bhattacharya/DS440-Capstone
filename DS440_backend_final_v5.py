import numpy as np
import time
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import math
import os
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
from random import choice
import imageio.v2 as imageio

# Load the basketball court image
path1 = r"C:/Users/Ivan9/OneDrive/桌面/School/DS440/basketball_court_half.png"
court_img = mpimg.imread(path1)

# Probabilities Dictionary
probabilities = {
    "shoot_success_close": 0.88,
    "shoot_success_mid": 0.8,
    "shoot_success_far": 0.69,
    "defender_influence_close": 0.65,
    "pass_intercept": 0.25,
    "ball_lost": 0.08,
    "blow_by": 0.3
}

# Function to update probabilities based on user-selected settings
def update_probabilities(settings):
    global probabilities

    if settings["shooting"] == "Easy Shooting":
        probabilities["shoot_success_close"] += 0.05
        probabilities["shoot_success_mid"] += 0.1
        probabilities["shoot_success_far"] += 0.12
        probabilities["defender_influence_close"] -= 0.1
    elif settings["shooting"] == "Difficult Shooting":
        probabilities["shoot_success_close"] -= 0.2
        probabilities["shoot_success_mid"] -= 0.2
        probabilities["shoot_success_far"] -= 0.1
        probabilities["defender_influence_close"] += 0.2

    if settings["handles"] == "Easy Handles":
        probabilities["blow_by"] += 0.2
        probabilities["ball_lost"] = 0.01  # Lower ball loss chance
    elif settings["handles"] == "Difficult Handles":
        probabilities["blow_by"] -= 0.1
        probabilities["ball_lost"] = 0.1  # Higher ball loss chance

    if settings["passing"] == "Easy Passing":
        probabilities["pass_intercept"] -= 0.1
    elif settings["passing"] == "Hard Passing":
        probabilities["pass_intercept"] += 0.1

# Define DQN
class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(4, 128)  # state: (x, y, defender_x, defender_y)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 3)  # actions: move, shoot, pass

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define environment details, actions, and rewards
actions = ["move", "shoot", "pass"]
gamma = 0.95  # discount factor
epsilon = 1.0  # exploration-exploitation tradeoff
epsilon_decay = 0.995
min_epsilon = 0.01
alpha = 0.01  # learning rate

# Initialize DQN and optimizer
model = DQN()
optimizer = optim.Adam(model.parameters(), lr=alpha)
loss_fn = nn.MSELoss()

# Experience replay buffer
replay_buffer = deque(maxlen=1000)

# Player and Defender initialization
num_players = 5
num_defenders = 5
player_positions = [(-20 + 10 * i, 20) for i in range(num_players)]
defender_positions = [(5 * i - 10, 15) for i in range(num_defenders)]
ball_position = player_positions[0]
ball_holder = 0

# Man coverage assignments
man_assignments = {i: i for i in range(num_defenders)}

# Team statistics
team_stats = {
    'shots_taken': 0,
    'shots_made': 0,
    'shots_missed': 0,
    'ball_lost': 0,
    'passes_attempted': 0,
    'passes_completed': 0,
    'passes_intercepted': 0,
    'total_points': 0,
    'three_pointers': 0,
    'two_pointers': 0
}

def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2)

def calculate_shot_success(pos, defender_pos):
    dist_to_hoop = distance(pos, (0, 5.2))
    dist_to_defender = distance(pos, defender_pos)

    # Base success probability based on distance to hoop
    if dist_to_hoop <= 10:
        base_prob = probabilities["shoot_success_close"]
    elif dist_to_hoop <= 23.75:
        base_prob = probabilities["shoot_success_mid"]
    else:
        base_prob = probabilities["shoot_success_far"]

    # Defender influence
    if dist_to_defender < 3:
        defender_factor = probabilities["defender_influence_close"]
    elif dist_to_hoop > 23.75:
        defender_factor = 0.85
    else:
        defender_factor = 1.0

    return base_prob * defender_factor

def move_toward(positions, idx, target):
    curr_x, curr_y = positions[idx]
    target_x, target_y = target

    # Calculate direction
    dx = target_x - curr_x
    dy = target_y - curr_y
    dist = math.sqrt(dx*dx + dy*dy)

    if dist > 0:
        # Normalize and scale movement
        step_size = 3.0
        dx = (dx/dist) * step_size
        dy = (dy/dist) * step_size

        # Update position within bounds
        new_x = clamp(curr_x + dx, -25, 25)
        new_y = clamp(curr_y + dy, 0, 47)
        positions[idx] = (new_x, new_y)

        # Update ball position if this player has the ball
        global ball_position
        if idx == ball_holder:
            ball_position = positions[idx]

def move_defenders():
    new_positions = []
    for i in range(num_defenders):
        target_player = man_assignments[i]
        target_pos = player_positions[target_player]
        curr_pos = defender_positions[i]

        # Calculate direction to assigned player
        dx = target_pos[0] - curr_pos[0]
        dy = target_pos[1] - curr_pos[1]
        dist = math.sqrt(dx*dx + dy*dy)

        if dist > 0:
            # Normalize and scale movement
            step_size = 2.0
            dx = (dx/dist) * step_size
            dy = (dy/dist) * step_size

            # Proposed new position
            new_x = clamp(curr_pos[0] + dx, -25, 25)
            new_y = clamp(curr_pos[1] + dy, 0, 47)

            # Check for collisions with other defenders
            collision = any(distance((new_x, new_y), pos) < 2.0 for pos in new_positions)

            if collision:
                # Move around by adding an angle offset
                angle = math.pi/4  # 45 degrees
                rot_dx = dx*math.cos(angle) - dy*math.sin(angle)
                rot_dy = dx*math.sin(angle) + dy*math.cos(angle)
                new_x = clamp(curr_pos[0] + rot_dx, -25, 25)
                new_y = clamp(curr_pos[1] + rot_dy, 0, 47)

            new_positions.append((new_x, new_y))
        else:
            new_positions.append(curr_pos)

    # Update all defender positions
    for i in range(num_defenders):
        defender_positions[i] = new_positions[i]

def select_action(state):
    if np.random.rand() < epsilon:
        return np.random.choice(actions)
    else:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            q_values = model(state_tensor)
            return actions[torch.argmax(q_values).item()]

def train_model(batch_size=32):
    if len(replay_buffer) < batch_size:
        return

    batch = np.random.choice(len(replay_buffer), batch_size, replace=False)
    for idx in batch:
        state, action, reward_value, next_state = replay_buffer[idx]

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0)

        with torch.no_grad():
            next_q_values = model(next_state_tensor)
            target = reward_value + gamma * torch.max(next_q_values)

        current_q_values = model(state_tensor)
        current_q_value = current_q_values[0, actions.index(action)]

        loss = loss_fn(current_q_value, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def clamp(value, min_value, max_value):
    return max(min_value, min(value, max_value))




def is_ball_held():
    # Check if any player is holding the ball (distance threshold < 1.0)
    return any(distance(player_positions[i], ball_position) < 1.0 for i in range(num_players))


def visualize_training(step, result_text, settings):
    plt.figure(figsize=(12, 8))
    plt.imshow(court_img, extent=[-25, 25, 0, 47])

    # Plot players and defenders
    for i, pos in enumerate(player_positions):
        color = 'cyan' if i == ball_holder else 'blue'
        plt.scatter(pos[0], pos[1], color=color, s=200, marker='o', edgecolors='black')

    for i, pos in enumerate(defender_positions):
        plt.scatter(pos[0], pos[1], color='red', s=200, marker='o', edgecolors='black')

    # Plot hoop and ball position
    plt.scatter(0, 5.2, color='orange', s=100, marker='o', edgecolors='black')
    plt.scatter(ball_position[0], ball_position[1], color='yellow', s=100, marker='x')

    # Calculate shooting proportions using actual stats
    total_shots_attempted = team_stats['shots_taken']
    three_pointers_attempted = team_stats['three_pointers'] + (team_stats['shots_missed'] * 0.4)  # Assume 40% of misses were 3pt attempts
    two_pointers_attempted = total_shots_attempted - three_pointers_attempted

    three_point_percentage = (
        f"{team_stats['three_pointers']}/{int(three_pointers_attempted)} "
        f"({team_stats['three_pointers'] / max(1, three_pointers_attempted) * 100:.2f}%)"
    )
    two_point_percentage = (
        f"{team_stats['two_pointers']}/{int(two_pointers_attempted)} "
        f"({team_stats['two_pointers'] / max(1, two_pointers_attempted) * 100:.2f}%)"
    )
    total_shooting_percentage = (
        f"{team_stats['shots_made']}/{team_stats['shots_taken']} "
        f"({team_stats['shots_made'] / max(1, team_stats['shots_taken']) * 100:.2f}%)"
    )

    # Display cumulative team stats and settings
    stats_text = (
        f"Team Statistics (Step {step + 1})\n"
        f"Total Points: {team_stats['total_points']}\n"
        f"Three Pointers Made: {three_point_percentage}\n"
        f"Two Pointers Made: {two_point_percentage}\n"
        f"Shooting Proportion: {total_shooting_percentage}\n"
        f"Passing: Attempted {team_stats['passes_attempted']}, Completed {team_stats['passes_completed']}, Intercepted {team_stats['passes_intercepted']}\n"
        f"Ball Lost: {team_stats['ball_lost']}\n"
        f"Last Action: {result_text}\n\n"
        f"Current Settings:\n"
        f"Shooting: {settings['shooting']}\n"
        f"Handles: {settings['handles']}\n"
        f"Passing: {settings['passing']}"
    )

    plt.text(26, 40, stats_text, fontsize=10, bbox=dict(facecolor='white', alpha=0.8))
    plt.title(f"Basketball RL Training - Step {step + 1}")
    plt.tight_layout()

    # Save and resize the plot to a consistent size
    temp_path = f'frame_{step}.png'
    plt.savefig(temp_path, bbox_inches='tight')
    plt.close()

    img = cv2.imread(temp_path)
    resized_img = cv2.resize(img, (960, 800))  # Resize to (960, 800) for consistency
    cv2.imwrite(temp_path, resized_img)  # Overwrite with resized image






def train_and_generate_video(settings):
    update_probabilities(settings)

    steps = 500
    print("Starting training...")

    for step in range(steps):
        print(f"Training progress: Step {step + 1}/{steps}", end='\r')

        global ball_position, ball_holder

        ball_holder = 0
        ball_position = player_positions[0]

        state = player_positions[ball_holder] + defender_positions[man_assignments[ball_holder]]
        result_text = ""

        terminal_event_occurred = False
        iteration_count = 0  # Track iterations within a step
        max_iterations = 50  # Set a maximum limit for iterations per step

        while not terminal_event_occurred:
            iteration_count += 1

            # Enforce action if maximum iterations are exceeded
            if iteration_count > max_iterations:
                action = "move"  # Default to move action
                result_text = "Default action triggered (move)"
            else:
                action = select_action(state)

            reward = 0  # Initialize reward for the current action

            if action == "move":
                if ball_holder is not None:
                    target = (np.random.uniform(-25, 25), np.random.uniform(0, 47))
                    old_pos = player_positions[ball_holder]
                    move_toward(player_positions, ball_holder, target)

                    defender_idx = man_assignments[ball_holder]
                    dist_to_defender = distance(player_positions[ball_holder], defender_positions[defender_idx])

                    # Check for blow-by based on probability
                    if dist_to_defender < 3:  # Only check if close to the defender
                        if np.random.random() < probabilities["blow_by"]:
                            result_text = "Blow-by successful!"
                            reward = 2  # Positive reward for evading the defender
                        else:
                            result_text = "Defender stops the move"
                            reward = -1  # Negative reward for being stopped
                    else:
                        result_text = "Moved past defender"
                        reward = 1  # Positive reward for moving without interference

                    # Simulate a chance to drop the ball during movement
                    if np.random.random() < probabilities["ball_lost"]:
                        ball_holder = None  # No one holds the ball
                        result_text = "Ball Lost"
                        ball_position = (np.random.uniform(-25, 25), np.random.uniform(0, 47))  # Randomly drop the ball
                        team_stats['ball_lost'] += 1
                        terminal_event_occurred = True

            elif action == "shoot":
                if ball_holder is not None:  # Ensure the ball is being held
                    team_stats['shots_taken'] += 1
                    defender_idx = man_assignments[ball_holder]
                    shot_success = calculate_shot_success(player_positions[ball_holder], defender_positions[defender_idx])

                    if np.random.random() < shot_success:
                        dist_to_hoop = distance(player_positions[ball_holder], (0, 5.2))
                        if dist_to_hoop > 23.75:
                            points = 3
                            team_stats['three_pointers'] += 1
                            result_text = "Made 3-pointer!"
                            reward = 5  # Higher reward for 3-pointer
                        else:
                            points = 2
                            team_stats['two_pointers'] += 1
                            result_text = "Made 2-pointer!"
                            reward = 2  # Reward for 2-pointer
                        team_stats['shots_made'] += 1
                        team_stats['total_points'] += points
                    else:
                        team_stats['shots_missed'] += 1
                        result_text = "Shot Missed"
                        reward = -1  # Negative reward for missing a shot

                    terminal_event_occurred = True
                else:
                    result_text = "Ball Not Held, Shot Failed"
                    reward = -3  # Strong negative reward for attempting to shoot without the ball

            elif action == "pass":
                if ball_holder is not None:
                    team_stats['passes_attempted'] += 1
                    new_holder = choice([i for i in range(num_players) if i != ball_holder])
                    intercepted = False

                    pass_path = np.array(player_positions[new_holder]) - np.array(player_positions[ball_holder])
                    pass_dist = np.linalg.norm(pass_path)

                    for def_pos in defender_positions:
                        def_vec = np.array(def_pos) - np.array(player_positions[ball_holder])
                        if np.linalg.norm(def_vec) < pass_dist:
                            proj = np.dot(def_vec, pass_path) / pass_dist
                            if 0 < proj < pass_dist:
                                perp_dist = np.linalg.norm(def_vec - proj * pass_path / pass_dist)
                                if perp_dist < 0.25 and np.random.random() < probabilities["pass_intercept"]:
                                    intercepted = True
                                    break



                    if intercepted:
                        team_stats['passes_intercepted'] += 1
                        result_text = "Pass Intercepted"
                        reward = -2  # Negative reward for intercepted pass
                        terminal_event_occurred = True
                    else:
                        team_stats['passes_completed'] += 1
                        ball_holder = new_holder
                        ball_position = player_positions[ball_holder]
                        result_text = "Pass Completed"
                        reward = 0  # Positive reward for successful pass
                else:
                    result_text = "Ball Not Held, Pass Failed"
                    reward = -3  # Strong negative reward for attempting to pass without the ball








            elif ball_holder is None and np.random.random() < probabilities["ball_lost"]:
                team_stats['ball_lost'] += 1
                result_text = "Ball Lost"
                reward = -3  # Strong negative reward for losing the ball
                terminal_event_occurred = True

            move_defenders()

            # Handle next state based on ball possession
            if ball_holder is not None:
                next_state = player_positions[ball_holder] + defender_positions[man_assignments[ball_holder]]
            else:
                next_state = ball_position + defender_positions[man_assignments[0]]  # Default to first defender's assignment
            
            replay_buffer.append((state, action, reward, next_state))  # Append reward to buffer
            state = next_state

        visualize_training(step, result_text, settings)

    print("\nTraining completed!")


    # Create the video after training
    print("Creating training video with imageio...")
    frame_paths = [f'frame_{i}.png' for i in range(steps)]
    frames = [imageio.imread(frame_path) for frame_path in frame_paths if os.path.exists(frame_path)]
    output_path = 'training_timelapse-enhancedv2.mp4'
    imageio.mimsave(output_path, frames, fps=1)

    print("\nTraining time-lapse video saved successfully.")

    # Clean up frame files after video creation
    for frame_path in frame_paths:
        if os.path.exists(frame_path):
            os.remove(frame_path)



if __name__ == "__main__":
    user_settings = {
        "shooting": "Default Shooting",
        "handles": "Default Handles",
        "passing": "Default Passing"
    }
    train_and_generate_video(user_settings)