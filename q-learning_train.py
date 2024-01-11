import numpy as np
from ElectricCarEnv import ElectricCarEnv
from algorithms import QLearningAgent

# Environment and Agent Initialization
env = ElectricCarEnv()
state_bins = [np.linspace(0, 50, 10), np.arange(0, 25), np.array([0, 1])]  # Discretize battery level, time, availability
action_bins = np.linspace(-25, 25, 10)  # Discretize actions (buy/sell amounts)
agent = QLearningAgent(state_bins, action_bins)

# Training Loop
num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    done = False
    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        if not done:
            agent.update(state, action, reward, next_state)
        state = next_state

# Save the Q-table
np.save('models/q_table.npy', agent.q_table)

