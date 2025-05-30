import numpy as np
import itertools

class QLearning_Lane_1Vehicle:
    def __init__(self, n_distance_bins=10, n_speed_bins=10, n_actions=5, alpha=0.1, gamma=0.9, epsilon=0.9):
        # Discretize distance features into bins
        self.distance_bins = np.linspace(0, 50, n_distance_bins)
        self.speed_bins = np.linspace(0, 10, n_speed_bins)

        self.actions = list(range(n_actions))  # Idle, left, right, accelerate, decelerate

        # Hyperparameters
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

        # Initialize Q-table with zeros
        # Dimensions: distance_x_bins * left_lane_bins * right_lane_bins * 2 (getting_closer)
        self.q_table = np.zeros((n_distance_bins, n_speed_bins, n_actions))

        # print(self.q_table)

        print(self.distance_bins)

        all_states = list(itertools.product(self.distance_bins, self.speed_bins))

        # Convert to a numpy array if needed
        all_states_array = np.array(all_states)

        print("Total number of states:", len(all_states))
        # print("All possible states:\n", all_states_array)

    def discretize_state(self, distance, speed):
        # Discretize continuous distances into bins
        dist_bin = np.digitize(distance, self.distance_bins) - 1
        speed_bin = np.digitize(speed, self.speed_bins) - 1

        # Clip to ensure values fall within range
        dist_bin = np.clip(dist_bin, 0, len(self.distance_bins) - 1)
        speed_bin = np.clip(speed_bin, 0, len(self.speed_bins) - 1)

        # Return tuple representing discrete state
        return (
            dist_bin, speed_bin)

    def choose_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.actions)  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def get_best_action_for_state_excluding_zero(self, state):
        # Get Q-values for all actions in the specified state
        action_values = self.q_table[state]

        # Exclude action `0` and find t.he action with the highest Q-value
        best_action = np.argmax(action_values[1:]) + 1  # +1 adjusts index since we skip action 0

        # Return the best action and its Q-value
        return best_action, action_values[best_action]

    def get_best_action_for_state_excluding_two(self, state):
        # Get Q-values for all actions in the specified state
        action_values = self.q_table[state]

        # Exclude action `2` by setting its value to a very low number (e.g., -inf)
        action_values_excluding_two = np.copy(action_values)
        action_values_excluding_two[2] = -np.inf

        # Find the action with the highest Q-value excluding action `2`
        best_action = np.argmax(action_values_excluding_two)

        # Return the best action and its Q-value
        return best_action, action_values[best_action]

    def update_q_value(self, state, action, reward, next_state):
        best_next_action = np.argmax(self.q_table[next_state])
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        td_error = td_target - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error

    def learn_v2(self, state, action, reward, next_state):
        self.update_q_value(state, action, reward, next_state)

    #def learn(self, distance, speed, action,
              #reward, next_distance, next_speed):
        # Discretize current and next states
        #state = self.discretize_state(distance, speed)
        #next_state = self.discretize_state(next_distance, next_speed)

        # Update Q-table
        #self.update_q_value(state, action, reward, next_state)

    def learn(self,state,action,reward,next_state):
        self.update_q_value(state, action, reward, next_state)
    def save_model(self, file_path):
        # Save the Q-table to a file
        np.save(file_path, self.q_table)
        print(f"Q-table saved to {file_path}")

    def load_model(self, file_path):
        # Load the Q-table from a file
        self.q_table = np.load(file_path)
        print(f"Q-table loaded from {file_path}")


class QLearningSemantic_1Vehicle(QLearning_Lane_1Vehicle):
    def __init__(self, n_distance_bins=10, n_speed_bins=10, n_actions=5, alpha=0.1, gamma=0.9, epsilon=0.9):
        # Initialize the parent class
        super().__init__(n_distance_bins, n_speed_bins, n_actions, alpha, gamma, epsilon)

    def get(self,speed, distance):
        """
        Classify the speed and distance into discrete states.

        Parameters:
        - speed (float): The speed of the vehicle.
        - distance (float): The distance to the closest vehicle.

        Returns:
        - tuple: A state representing the speed category and distance category.
        """
        state = None

        dict_state_dist={
            "very close" : 0,
            "close" : 1,
            "moderate" :2,
            "far":3
        }


        dict_state_speed={
            "very slow" : 0,
            "slow" : 1,
            "medium" :2,
            "fast":3,
            "very fast" :4
        }

        if distance < 10:
            distance_state = "very close"
        elif 10 <= distance < 25:
            distance_state = "close"
        elif 25 <= distance < 50:
            distance_state = "moderate"
        else:
            distance_state = "far"

        if 0 < speed < 5:
            speed_state = "very slow"

        elif 5 < speed < 10:
            speed_state = "slow"

        elif 10 <= speed <= 16:
            speed_state = "medium"
        elif 16 < speed <= 21:
            speed_state = "fast"
        #elif 20 <= speed <= 30:
        else:
            speed_state = "very fast"
        #else:
            #speed_state = "unknown"

        #if speed_state != "unknown":
        state = (speed_state, distance_state)


        state_integer = (dict_state_speed[speed_state],dict_state_dist[distance_state]  )

        return state,state_integer


#q1 = QLearningSemantic_1Vehicle(5,4)
#state,state_integer =q1.get(20,10)


#print("state ",state)
#print("state_integer ",state_integer)

#q1.q_table[state_integer][3] = -100
#print(q1.q_table[state_integer])

