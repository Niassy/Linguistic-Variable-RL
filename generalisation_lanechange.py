"""
04/11/2024
introducing q learning

from env 12

# 04/12/2024 ( 1 month already wow :(   )

- In this scenario,the agent has to know how to accelerate and decelerate
- No changing lane

objective
- Use our approach of previous state action influence
-

"""

import gymnasium as gym
import highway_env

import matplotlib.pyplot as plt

import numpy as np

from dummu_q import QLearningSemantic_1Vehicle

#from fuzzy_logic import FuzzyCollision

#fuzzy = FuzzyCollision()

actions_dict = {

'LANE_LEFT' :0,
  'IDLE' : 1,
  'LANE_RIGHT':2,
'FASTER':3,
'SLOWER':4
 }

print(gym.envs.registry.keys())

config = {

    "screen_width": 640,
    "screen_height": 480,
    "vehicles_count": 1,
    "observation": {
        "type": "Kinematics",
        "vehicles_count": 1,
        # "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"
        "features": ["presence", "x", "y", "vx", "vy"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20]
        },
        "absolute": False,
        "order": "sorted"
    }
}

env = gym.make(
    "highway-v0",
    render_mode='rgb_array',
    config=config
)

# Constants
C = 100  # Collision penalty
alpha = 1  # Scaling factor for distance difference


# Reward function
def reward_function(current_distance, collided, previous_distance):

    # Collision penalty
    r_collision = -C if collided else 0

    # Distance maintenance reward (difference between current and previous distance)
    r_distance = 0
    if previous_distance is not None:
        r_distance = alpha * (current_distance - previous_distance)

    # Total reward
    r_total = r_collision + r_distance
    return r_total

def get_reward(d,speed,collided,previous_d,previous_speed):

    if collided:
        return -10

    elif previous_d - d >0:
        return 0.1

    return 0

def get_linguistic_label(reward):
    if reward < -5:
        return "Low"
    elif -5 <= reward <= 10:
        return "Medium"
    else:
        return "High"
    
    
q_semantic = QLearningSemantic_1Vehicle(5,4)

dummy_state = (2, 4, 4)
dummy_state2 = (3, 4, 2)

action = 1
done = False
collide = False

#x = obs[1, 1]
last_x = 0

# Initialize the previous distance
previous_distance = None

# Variables to store data across episodes
time_before_collision = []
average_distances = []
lane_changes = []
number_collisions = []

time_end = False

step_time = 20

lane_change_count = 0

dummy_state1 = (4, 3, 4)
dummy_state2 = (2, 4, 4)
dummy_state3 = (3, 4, 2)
dummy_state4 = (4, 4, 4)
dummy_state5 = (2, 2, 4)
dummy_state6 = (3, 3, 4)

previous = {
    "nearest_same": 10,
    "nearest_left": 10,
    "nearest_right": 10
}

total_distances = []

total_dist = 0

knn_count_per_episode = []
q_learning_count_per_episode = []

start_lane = -1
current_lane = -1
number_episode = 50
training_rewards = []

#Q.load_model("highenv_q_learning3.npy")
total_hit = 0

memory_array = []

episode_rewards = []

total_dist = 0

for episode in range(number_episode):

    time_ep = 15
    total_reward = 0

    done = False
    total_distance = 0
    steps = 0
    number_collision = 0

    diff_prev_dist = 0
    lane_change_count = 0

    #total_dist = 0

    total_reward = 0
    knn_count = 0
    q_learning_count = 0

    obs, info = env.reset()

    distance, speed_ego, speed_other = info["dist_x"], info["vehicle1_speed"], info["vehicle2_speed"]

    hit = False
    reward = 0
    while not done:

        action = 1

        state, state_integer = q_semantic.get(speed_ego, distance)

        #print("state ",state , " state_integer ",state_integer)

        action_q = np.argmax(q_semantic.q_table[state_integer])
        radical_action = 3

        action = radical_action

        if q_semantic.q_table[state_integer][radical_action] <0:
            action= action_q
        obs, _, done, truncated, info = env.step(action)

        if info.get("crashed", False):
            hit = True
            total_hit+=1
            reward = - 10
        total_reward += reward  # Accumulate reward


        next_distance,next_speed_ego,next_speed_other = (info["dist_x"], info["vehicle1_speed"],
                                                         info["vehicle2_speed"])

        next_state = q_semantic.get(next_distance, next_speed_ego)

        q_semantic.q_table[state_integer][action] = reward

        if done:
            if action == 2:
                action = 0

        distance, speed_ego = next_distance, next_speed_ego

        env.render()

        time_ep-=1

        if time_ep <=0:
            done = True
        if done:
            total_dist+=distance
    episode_rewards.append(total_reward)  # Store the total reward

#print("Hits = ",str(number_hit)+" / ",str(num_episode // 2))

print("Hits = ",str(total_hit)+" / ",str(number_episode))
print("total dist ",str(total_dist))
print("Average dist = ",str(total_dist /number_episode ))


plt.plot(episode_rewards)
plt.title("Rewards Over Episodes")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.show()

"""
linguistic_labels = [get_linguistic_label(r) for r in episode_rewards]


label_to_value = {"Low": 0, "Medium": 1, "High": 2}
y_values = [label_to_value[label] for label in linguistic_labels]

plt.figure(figsize=(10, 4))
plt.plot(y_values, marker='o')
plt.yticks([0, 1, 2], ["Low", "Medium", "High"])
plt.xlabel("Episode")
plt.ylabel("Reward Category")
plt.title("Linguistic Reward Classification per Episode")
plt.grid(True)
plt.tight_layout()
plt.show()
"""






