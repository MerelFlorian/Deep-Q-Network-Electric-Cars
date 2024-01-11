import numpy as np
import pandas as pd
from ElectricCarEnv import ElectricCarEnv
from algorithms import QLearningAgent

def validate_agent(env, agent, num_episodes):
    """"
    Function to validate the agent on a validation set.
    """
    total_rewards = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(state)  # Choose action based on policy
            state, reward, done, _ = env.step(action)
            total_rewards += reward
    
    average_reward = total_rewards / num_episodes
    return average_reward

# Environment and Agent Initialization
env = ElectricCarEnv()
state_bins = [np.linspace(0, 50, 10), np.arange(0, 25), np.array([0, 1])]  # Discretize battery level, time, availability
action_bins = np.linspace(-25, 25, 10)  # Discretize actions (buy/sell amounts)
agent = QLearningAgent(state_bins, action_bins)

# Load validation data into the environment
env.data = pd.read_csv('data/validate_clean.csv') 

# Create a new agent instance or use the existing one
test_agent = QLearningAgent(state_bins, action_bins) 

# Load the Q-table
test_agent.q_table = np.load('models/q_table.npy')

# Test the agent
num_test_episodes = 100  # Define the number of episodes for testing
test_performance = validate_agent(env, agent, num_test_episodes)

print(f"Average reward on validation set: {test_performance}")