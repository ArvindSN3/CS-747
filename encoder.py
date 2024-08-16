import argparse

def find_coordinates(state):
    b1_square = int(state[0:2]) - 1
    b2_square = int(state[2:4]) - 1
    r_square = int(state[4:6]) - 1 
    b1_x, b1_y = divmod(b1_square, 4)
    b2_x, b2_y = divmod(b2_square, 4)
    r_x, r_y = divmod(r_square, 4)

    return b1_x, b1_y, b2_x, b2_y, r_x, r_y

def decode_player_possession(state):
    if int(state) % 10 == 1:
        return 1
    else:
        return 2

def compute_next_state(x1, y1, x2, y2, x3, y3, ball_possession):
    # Define the 4x4 grid as a list of lists
    grid = [
        [1, 2, 3, 4],
        [5, 6, 7, 8],
        [9, 10, 11, 12],
        [13, 14, 15, 16]
    ]

    # Determine the square number for each player based on coordinates
    player1_square = grid[x1][y1]
    player2_square = grid[x2][y2]
    opponent_square = grid[x3][y3]

    # Construct the 7-digit string
    state_string = f"{player1_square:02d}{player2_square:02d}{opponent_square:02d}{ball_possession}"

    return state_string

def calculate_movement_tackling_transition(transition_prob, rewards, state, action, ball_possession, p, x1, y1, x2, y2, x3, y3, state_list, opponent_prob):
    num_states = len(state_list)
    loss_index = state_list.index("loss")

    if action <= 3:
        if action == 0:  
            new_x1, new_y1 = x1, y1 - 1
        elif action == 1:  
            new_x1, new_y1 = x1, y1 + 1
        elif action == 2:  
            new_x1, new_y1 = x1 - 1, y1
        elif action == 3:  
            new_x1, new_y1 = x1 + 1, y1

        if ball_possession == 1:  # B1 movement with the ball
            if not (0 <= new_x1 <= 3 and 0 <= new_y1 <= 3):
                # Player goes out of bounds, end the episode
                transition_prob[loss_index] = 1
                rewards[loss_index] = 0
            else:
                next_state = compute_next_state(new_x1, new_y1, x2, y2, x3, y3, ball_possession)
                next_state_index = state_list.index(next_state)

                if (abs(new_x1 - x3) + abs(new_y1 - y3) == 1) or (new_x1 == x3 and new_y1 == y3):
                    # Successful Tackle
                    transition_prob[next_state_index] = 0.5 - p
                    transition_prob[loss_index] = 0.5 + p
                    rewards[next_state_index] = 0
                    rewards[loss_index] = 0
                else:
                    transition_prob[next_state_index] = 1 - 2 * p
                    transition_prob[loss_index] = 2 * p
                    rewards[loss_index] = 0
                    rewards[next_state_index] = 0

        else:  # B1 movement without the ball
            if not (0 <= new_x1 <= 3 and 0 <= new_y1 <= 3):
                # Player goes out of bounds, end the episode
                transition_prob[loss_index] = 1
                rewards[loss_index] = 0
            else:
                next_state = compute_next_state(new_x1, new_y1, x2, y2, x3, y3, ball_possession)
                next_state_index = state_list.index(next_state)

                transition_prob[next_state_index] = 1 - p
                transition_prob[loss_index] = p
                rewards[loss_index] = 0
                rewards[next_state_index] = 0

    else:
        if action == 4:  
            new_x2, new_y2 = x2, y2 - 1
        elif action == 5:  
            new_x2, new_y2 = x2, y2 + 1
        elif action == 6:  
            new_x2, new_y2 = x2 - 1, y2
        elif action == 7:  
            new_x2, new_y2 = x2 + 1, y2

        if ball_possession == 2:  # B2 movement with the ball
            if not (0 <= new_x2 <= 3 and 0 <= new_y2 <= 3):
                # Player goes out of bounds, end the episode
                transition_prob[loss_index] = 1
                rewards[loss_index] = 0

            else:
                next_state = compute_next_state(x1, y1, new_x2, new_y2, x3, y3, ball_possession)
                next_state_index = state_list.index(next_state)

                if (abs(new_x2 - x3) + abs(new_y2 - y3) == 1) or (new_x2 == x3 and new_y2 == y3):
                    # Successful Tackle
                    transition_prob[next_state_index] = 0.5 - p
                    transition_prob[loss_index] = 0.5 + p
                    rewards[next_state_index] = 0
                    rewards[loss_index] = 0
                else:
                    transition_prob[next_state_index] = 1 - 2 * p
                    transition_prob[loss_index] = 2 * p
                    rewards[loss_index] = 0
                    rewards[next_state_index] = 0

        else:  # B2 movement without the ball
            if not (0 <= new_x2 <= 3 and 0 <= new_y2 <= 3):
                # Player goes out of bounds, end the episode
                transition_prob[loss_index] = 1
                rewards[loss_index] = 0
            else:
                next_state = compute_next_state(x1, y1, new_x2, new_y2, x3, y3, ball_possession)
                next_state_index = state_list.index(next_state)

                transition_prob[next_state_index] = 1 - p
                transition_prob[loss_index] = p
                rewards[loss_index] = 0
                rewards[next_state_index] = 0

    # Multiply transition probabilities by opponent's action probability
    for i in range(len(transition_prob)):
        transition_prob[i] *= opponent_prob

    return transition_prob, rewards

# Rest of the code
def calculate_pass_transition(transition_prob, rewards, state, action, ball_possession, p, q, x1, y1, x2, y2, x3, y3, state_list, opponent_prob):
    num_states = len(state_list)
    loss_index = state_list.index("loss")
    
    # Calculate the distance between B1 and B2
    distance = max(abs(x1 - x2), abs(y1 - y2))
    
    # Calculate the probability of a successful pass
    pass_success_prob = q - 0.1 * distance
    
    # If opponent (R) is in between B1 and B2
    if (x1 < x3 < x2 or x1 > x3 > x2) and (y1 < y3 < y2 or y1 > y3 > y2):
        pass_success_prob /= 2
    
    # Calculate the next state if the pass is successful
    next_state = compute_next_state(x1, y1, x2, y2, x3, y3, ball_possession)
    next_state_index = state_list.index(next_state)
    
    # Update transition probabilities and rewards for the new state
    transition_prob[next_state_index] = pass_success_prob
    transition_prob[loss_index] = 1 - pass_success_prob
    rewards[loss_index] = 0
    rewards[next_state_index] = 0  # No immediate reward
    
    # Multiply transition probabilities by opponent's action probability
    for i in range(len(transition_prob)):
        transition_prob[i] *= opponent_prob
    
    return transition_prob, rewards

def calculate_opponent_state(opponent_action, x_b1, y_b1, x_b2, y_b2, x_r, y_r, ball_possession):
    # Calculate the new positions of B1, B2, and R after the opponent's action
    if opponent_action == "L":
        new_x_r, new_y_r = x_r, y_r - 1
    elif opponent_action == "R":
        new_x_r, new_y_r = x_r, y_r + 1
    elif opponent_action == "U":
        new_x_r, new_y_r = x_r - 1, y_r
    elif opponent_action == "D":
        new_x_r, new_y_r = x_r + 1, y_r

    # Determine the resulting state after the opponent's move
    next_state = compute_next_state(x_b1, y_b1, x_b2, y_b2, new_x_r, new_y_r, ball_possession)

    return next_state, new_x_r, new_y_r

def print_transitions(state_list, opponent_transitions, p, q):
    num_states = len(state_list)
    num_actions = 10

    for s1 in range(num_states - 2):
        state = state_list[s1]
        ball_possession = decode_player_possession(state)
        x_b1, y_b1, x_b2, y_b2, x_r, y_r = find_coordinates(state)

        for opponent_action, opponent_prob in opponent_transitions[state].items():
            # Initialize the transition probabilities
            transition_prob = [0.0] * num_states
            rewards = [0.0] * num_states

            next_opponent_state, new_x_r, new_y_r = calculate_opponent_state(opponent_action, x_b1, y_b1, x_b2, y_b2, x_r, y_r, ball_possession)

            for action in range(num_actions):
                if action <= 7:  # Movement actions
                    if action <= 3:  # B1 movement
                        transition_prob, rewards = calculate_movement_tackling_transition(transition_prob, rewards,
                            next_opponent_state, action, ball_possession, p, x_b1, y_b1, x_b2, y_b2, new_x_r, new_y_r, state_list, opponent_prob)
                    else:  # B2 movement
                        transition_prob, rewards = calculate_movement_tackling_transition(transition_prob, rewards,
                            next_opponent_state, action, ball_possession, p, x_b1, y_b1, x_b2, y_b2, new_x_r, new_y_r, state_list, opponent_prob)
                elif action == 8:  # Passing action
                    transition_prob, rewards = calculate_pass_transition(transition_prob, rewards,
                        next_opponent_state, action, ball_possession, p, q, x_b1, y_b1, x_b2, y_b2, new_x_r, new_y_r, state_list, opponent_prob)
                elif action == 9:  # Shooting action
                    loss_index = state_list.index("loss")
                    goal_index = state_list.index("win")

                    if ball_possession == 1:
                        x_shooter = x_b1
                        y_shooter = y_b1
                    else:
                        x_shooter = x_b2
                        y_shooter = y_b2

                    r_in_front_of_goal = (new_x_r in [1, 2]) and (new_y_r in [3])

                    if r_in_front_of_goal:
                        probability_of_goal = (q - 0.2 * (3 - x_shooter)) / 2
                    else:
                        probability_of_goal = q - 0.2 * (3 - x_shooter)

                    # If the shot is successful (goal)
                    transition_prob[goal_index] = probability_of_goal
                    rewards[goal_index] = 1  # Goal scored

                    # If the shot fails
                    transition_prob[loss_index] = 1 - probability_of_goal
                    rewards[loss_index] = 0

                    # Multiply transition probabilities by opponent's action probability
                    for i in range(len(transition_prob)):
                        transition_prob[i] *= opponent_prob

            # Print the transition probabilities and rewards for the current state-action pair
            for next_state_index, reward, prob in zip(range(num_states), rewards, transition_prob):
                if prob > 0:
                    print("transition", s1, action, next_state_index, reward, prob)

def encode_mdp(opponent_policy, p, q):
    states = []
    state_list = []
    num_states = 0

    # Create states
    opponent_actions = {"L", "R", "U", "D"}  # Opponent actions
    state_data = open(opponent_policy).read().strip().split("\n")
    opponent_transitions = {}

    for line in state_data[1:]:  # Skip the header line
        state = line.split()[0]
        state_list.append(state)
        opponent_transitions[state] = {action: float(prob) for action, prob in zip(opponent_actions, line.split()[1:])}

    state_list.append("loss")
    state_list.append("win")
    num_actions = 10  
    num_states = len(state_list)
    print("numStates", num_states)
    print("numActions", num_actions)
    print("end", num_states - 1, num_states)
    print_transitions(state_list, opponent_transitions, p, q)
    print("mdptype", "episodic")
    print("discount", 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--opponent", required=True)
    parser.add_argument("--p", type=float, required=True)
    parser.add_argument("--q", type=float, required=True)
    args = parser.parse_args()

    opponent_policy_file = args.opponent
    p = args.p
    q = args.q

    encode_mdp(opponent_policy_file, p, q)