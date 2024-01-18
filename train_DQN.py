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
        """
        Function to train the neural network on a batch of samples from memory.
        """
        if len(self.memory) < self.batch_size:
            return  # Do not train if not enough samples

        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract information from each memory and convert to numpy arrays
        # Ensure the array is of type float for subsequent operations
        states = np.array([m[0] for m in minibatch], dtype=np.float32)
        actions = np.array([m[1] for m in minibatch], dtype=np.int64).reshape(-1, 1)
        rewards = np.array([m[2] for m in minibatch], dtype=np.float32)
        next_states = np.array([m[3] for m in minibatch], dtype=np.float32)
        dones = np.array([m[4] for m in minibatch], dtype=np.float32)

        # Normalize states and next states
        states[:, 0] /= 50.0  # Make sure to use float division
        states[:, 1] /= 24.0  # Make sure to use float division
        next_states[:, 0] /= 50
        next_states[:, 1] /= 24

        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute Q values for current states
        Q_expected = self.model(states).gather(1, actions)

        # Compute Q values for next states and calculate target
        with torch.no_grad():
            Q_next = self.model(next_states).max(1)[0].detach()
            Q_target = rewards + (self.gamma * Q_next * (1 - dones))

        # Calculate loss
        loss = nn.MSELoss()(Q_expected, Q_target.unsqueeze(1))

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


def train_DQN(env, agent):
    """
    Function to train the DQN agent.
    """
    num_episodes = 1

    for episode in range(num_episodes):
        state = env.reset()
        done = False

        if not isinstance(state, np.ndarray) or state.shape != (state_size,):
            state = np.reshape(state, (state_size,))  # Ensure the state has the correct shape

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            if not isinstance(next_state, np.ndarray) or next_state.shape != (state_size,):
                next_state = np.reshape(next_state, (state_size,)) 
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            if len(agent.memory) > agent.batch_size:
                agent.replay()

        print(f"Episode {episode}/{num_episodes} completed.")

    agent.save("dqn_model.pth")

if __name__ == "__main__":
    env = ElectricCarEnv()
    state_size = 3  
    action_size = 5000  
    agent = DQNAgent(state_size, action_size)
    
    train_DQN(env, agent)
