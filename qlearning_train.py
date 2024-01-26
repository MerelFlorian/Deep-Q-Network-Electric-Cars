import numpy as np
from ElectricCarEnv import Electric_Car
from algorithms import QLearningAgent
import matplotlib.pyplot as plt
import pandas as pd

def validate_agent(test_env, test_agent, qtable):
    """"
    Function to validate the agent on a validation set.
    """
    state = test_env.reset()
    done = False
    total_rewards = 0
    test_agent.q_table = qtable
    while not done:
        action = test_agent.choose_action(state)  # Choose action based on policy
        state, reward, done, _, _ = test_env.step(action)
        total_rewards += reward

    return total_rewards
  
# Environment and Agent Initialization
env = Electric_Car("./data/train.xlsx", "./data/f_train.xlsx")
state_bins = [np.linspace(0, 50, 50), np.arange(0, 25), np.array([0, 1])]  # Discretize battery level, time, availability
action_bins = np.linspace(-1, 1, 50)  # Discretize actions (buy/sell amounts)
agent = QLearningAgent(state_bins, action_bins)

# Load validation data into the environment
test_env = Electric_Car("data/validate.xlsx", "data/f_val.xlsx")
test_agent = QLearningAgent(state_bins, action_bins) 

# Training Loop
num_episodes = 100
total_rewards = []
total_validation_rewards = []
highest_reward = -np.inf
best_battery_levels = 0
validation_rewards = []

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0
    battery_levels = []

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _, _ = env.step(action)
        if not done:
            agent.update(state, action, reward, next_state)
            battery_levels.append(env.battery_level)
        state = next_state
        total_reward += reward

    # Keep track of the total reward for this episode
    total_rewards.append(total_reward)

    # Run validation 
    validation_reward = validate_agent(test_env, test_agent, qtable=agent.q_table)
    total_validation_rewards.append(validation_reward)
    validation_rewards.append(validation_reward)

    # Check and update the highest reward and best episode
    if validation_reward > highest_reward:
        highest_reward = validation_reward
        best_episode = episode
        best_q_table = agent.q_table.copy()
        best_battery_levels = battery_levels.copy()

    print(f"Episode {episode} reward: {total_reward} | Validation reward: {validation_reward}")

# Save the best Q-table
np.save('models/best_q_table_2.npy', best_q_table)
print(f"Best Q-table saved from episode {episode} with total reward: {highest_reward}")