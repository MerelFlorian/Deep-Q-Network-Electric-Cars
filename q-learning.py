import gym
import numpy as np
import pandas as pd
from gym import spaces
from ElectricCarEnv import ElectricCarEnv

# Assuming ElectricCarEnv is defined as per your implementation.

# Helper function to discretize the state
def discretize_state(state, battery_bins, time_bins):
    battery_level, time_of_day, car_available = state
    battery_level_discrete = np.digitize(battery_level, battery_bins) - 1
    time_of_day_discrete = np.digitize(time_of_day, time_bins) - 1
    car_available_discrete = int(car_available)
    return (battery_level_discrete, time_of_day_discrete, car_available_discrete)

# Initialize environment
env = ElectricCarEnv()

# Discretization parameters
battery_bins = np.linspace(0, env.max_battery, 10)  # 10 battery level bins
time_bins = np.linspace(0, 24, 24)  # 24 hours

# Q-table dimensions
num_battery_levels = len(battery_bins)
num_time_of_day = len(time_bins)
num_car_available_states = 2  # Car available or not
num_actions = env.action_space.shape[0]

# Initialize Q-table
Q = np.zeros((num_battery_levels, num_time_of_day, num_car_available_states, num_actions))

# Learning parameters
learning_rate = 0.1
discount_factor = 0.99
epsilon = 0.1
num_episodes = 1000

# Training loop
for episode in range(num_episodes):
    state = env.reset()
    done = False

    while not done:
        # Discretize the state
        discrete_state = discretize_state(state, battery_bins, time_bins)

        # Epsilon-greedy action selection
        if np.random.random() < epsilon:
            action = env.action_space.sample()  # Explore
        else:
            action = np.argmax(Q[discrete_state])  # Exploit

        # Take action
        next_state, reward, done, _ = env.step(action)
        next_discrete_state = discretize_state(next_state, battery_bins, time_bins)

        # Update Q-table
        old_value = Q[discrete_state + (action,)]
        next_max = np.max(Q[next_discrete_state])
        new_value = (1 - learning_rate) * old_value + learning_rate * (reward + discount_factor * next_max)
        Q[discrete_state + (action,)] = new_value

        state = next_state
