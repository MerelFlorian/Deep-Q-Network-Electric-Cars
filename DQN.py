import numpy as np
import pandas as pd
import random
from collections import deque
import torch
import torch.nn as nn
import torch.optim as optim
from ElectricCarEnv import ElectricCarEnv

class QNetwork(nn.Module):
    """
    This class represents the neural network used to approximate the Q-function.
    """
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """
    This class represents the DQN agent.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.learning_rate = 0.001

        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        """
        Function to store a transition in memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state_vector = [state[0] / 50, state[1] / 24, state[2]]  # Assuming state is a list or array
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)

        if np.random.rand() > self.epsilon:  # Epsilon-greedy approach
            with torch.no_grad():
                action_values = self.model(state_tensor)
                return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.randrange(self.action_size)

    def replay(self):
        """Train the network using randomly sampled experiences from the memory."""
        if len(self.memory) < self.batch_size:
            return  # Do not train if not enough samples

        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            # Normalize and convert states
            state_vector = [state[0] / 50, state[1] / 24, state[2]]
            state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)
            next_state_vector = [next_state[0] / 50, next_state[1] / 24, next_state[2]]
            next_state_tensor = torch.FloatTensor(next_state_vector).unsqueeze(0)

            # Compute Q values for current state
            Q_values = self.model(state_tensor)

            # Compute Q values for next state and calculate target
            with torch.no_grad():
                next_Q_values = self.model(next_state_tensor)
                max_next_Q_values = next_Q_values.max(1)[0]
                target_Q_values = reward + (self.gamma * max_next_Q_values * (1 - int(done)))

            # Get the Q-value for the action taken
            Q_value = Q_values.squeeze(0)[action]

            # Calculate loss
            loss = nn.MSELoss()(Q_value, target_Q_values)

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        """
        Function to load the model's weights.
        """
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        """
        Function to save the model's weights.
        """
        torch.save(self.model.state_dict(), name)

env = ElectricCarEnv()
state_size = 3  
action_size = 5000  
agent = DQNAgent(state_size, action_size)

num_episodes = 1000
for episode in range(num_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    for time in range(500):  # or some other condition for an episode end
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            break
        if len(agent.memory) > agent.batch_size:
            agent.replay()
    print(f"Episode {episode}/{num_episodes} completed.")

agent.save("dqn_model.pth")
