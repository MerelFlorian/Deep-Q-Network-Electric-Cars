import numpy as np
from ElectricCarEnv import ElectricCarEnv
from algorithms import QLearningAgent
import matplotlib.pyplot as plt


# Environment and Agent Initialization
env = ElectricCarEnv()
state_bins = [np.linspace(0, 50, 50), np.arange(0, 25), np.array([0, 1])]  # Discretize battery level, time, availability
action_bins = np.linspace(-25, 25, 5000)  # Discretize actions (buy/sell amounts)
agent = QLearningAgent(state_bins, action_bins)

# Training Loop
num_episodes = 1000
total_rewards = []
highest_reward = -np.inf

for episode in range(num_episodes):
    state = env.reset()
    done = False
    total_reward = 0

    while not done:
        action = agent.choose_action(state)
        next_state, reward, done, _ = env.step(action)
        if not done:
            agent.update(state, action, reward, next_state)
        state = next_state
        total_reward += reward

    # Keep track of the total reward for this episode
    total_rewards.append(total_reward)
    print(f"Episode: {episode}, Total Reward: {total_reward}")

    # Check and update the highest reward and best episode
    if total_reward > highest_reward:
        highest_reward = total_reward
        best_episode = episode
        best_q_table = agent.q_table.copy()

# Save the best Q-table

np.save('models/best_q_table.npy', best_q_table)
print(f"Best Q-table saved from episode {episode} with total reward: {highest_reward}")


# Save the Q-table
np.save('models/q_table.npy', agent.q_table)

# Plot the learning progress
plt.plot(total_rewards)
plt.title('Learning Progress')
plt.xlabel('Episode')
plt.ylabel('Total Reward')
plt.show()
