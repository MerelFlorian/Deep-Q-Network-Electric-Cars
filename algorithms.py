import gym
import numpy as np
import pandas as pd
from gym import spaces
import random
from ElectricCarEnv import ElectricCarEnv

class QLearningAgent:
    """
    Implements a simple tabular Q-learning agent for the electric car trading problem.
    """	
    def __init__(self, state_bins, action_bins, learning_rate=0.1, discount_factor=0.99, epsilon=1.0, epsilon_decay=0.995, min_epsilon=0.01, max_battery=50):
        self.state_bins = state_bins
        self.action_bins = action_bins
        self.max_battery = max_battery
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros(shape=(len(state_bins[0]), len(state_bins[1]), len(state_bins[2]), len(action_bins)))

    def discretize_state(self, state):
        """	
        Discretizes the state into a tuple of indices.
        """
        battery, time, availability = state
        battery_idx = np.digitize(battery, self.state_bins[0]) - 1
        time_idx = np.digitize(time, self.state_bins[1]) - 1
        availability_idx = int(availability)
        return battery_idx, time_idx, availability_idx

    def discretize_action(self, action):
        """
        Discretizes the action into an index.
        """
        return np.digitize(action, self.action_bins) - 1

    def choose_action(self, state):
        """
        Chooses an action using epsilon-greedy.
        """
        # Ensure state is within valid range
        if not self.is_valid_state(state):
            return 0 #TODO decide what to do for invalid action
         
        if np.random.random() < self.epsilon:
            return random.choice(range(len(self.action_bins)))
        else:
            discretized_state = self.discretize_state(state)
            return np.argmax(self.q_table[discretized_state])

    def update(self, state, action, reward, next_state):
        """
        Updates the Q-table using Q-learning.
        """
        # Skip update if state is invalid
        if not self.is_valid_state(state) or not self.is_valid_state(next_state):
            return  
        
        discretized_state = self.discretize_state(state)
        discretized_next_state = self.discretize_state(next_state)
        discretized_action = self.discretize_action(action)
        best_next_action = np.argmax(self.q_table[discretized_next_state])
        td_target = reward + self.gamma * self.q_table[discretized_next_state + (best_next_action,)]
        td_error = td_target - self.q_table[discretized_state + (discretized_action,)]
        self.q_table[discretized_state + (discretized_action,)] += self.lr * td_error
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def is_valid_state(self, state):
        """"
        Checks if the state is valid.
        """
        # Implement logic to check if state is valid
        battery, time, availability = state
        return 0 <= battery <= self.max_battery and 1 <= time <= 24 and 0 <= availability <= 1
