import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.distributions import Categorical
from gym import Env
from ElectricCarEnv import ElectricCarEnv
import random
import numpy as np

class LSTM_PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, lstm_layers=2):
        super(LSTM_PolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(state_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc_mean = nn.Linear(hidden_size, action_size)
        self.fc_log_std = nn.Linear(hidden_size, action_size)
        

    def forward(self, state, hidden_state=None):
        # state shape: (batch, sequence, features)
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(state)
        else:
            lstm_out, hidden_state = self.lstm(state, hidden_state)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        action_mean = self.fc_mean(lstm_out)  # Linear layer for mean
        action_log_std = self.fc_log_std(lstm_out)  # Linear layer for log_std
        return action_mean, action_log_std.exp(), hidden_state


    def init_hidden(self, batch_size):
        # Initializes the hidden state
        # This depends on the number of LSTM layers and whether it's bidirectional
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_(),
                  weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_())
        return hidden
    
def normalize_rewards(rewards):
    rewards = np.array(rewards)
    mean = np.mean(rewards)
    std = np.std(rewards)
    if std == 0:
        return rewards - mean
    return (rewards - mean) / std
    
def compute_returns(rewards: list, gamma=0.99) -> torch.Tensor:
    """ Computes the discounted returns for a list of rewards.

    Args:
        rewards (list): The list of rewards.
        gamma (float, optional): The discount factor. Defaults to 0.99.

    Returns:
        list: The discounted returns.
    """
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns)


def train_policy_gradient(env: Env, policy_network: LSTM_PolicyNetwork, episodes=10, lr=0.0001, gamma=0.6, epsilon=0.2, sequence_length=12):
    """ Trains a policy network using the policy gradient algorithm.

    Args:
        env (Env): The environment to train on.
        policy_network (PolicyNetwork): The policy network to train.
        episodes (int, optional): The number of episodes to train for. Defaults to 10.
        lr (float, optional): The learning rate. Defaults to 0.01.
        gamma (float, optional): The discount factor. Defaults to 0.99.
        epsilon (float, optional): The probability of taking a random action. Defaults to 0.1.
    """
    # Initialize the optimizer
    optimizer = optim.Adam(policy_network.parameters(), lr=lr)
    batch_size = 1  # Assuming we're dealing with single episodes

    for episode in range(episodes):
        # Reset the environment
        state = env.reset()
        # Initialize the lists to store the states, rewards
        states, rewards, log_probs = [], [], []
        hidden_state = policy_network.init_hidden(batch_size)
        done = False

        while not done:
            # Prepare the sequence of states
            states.append(torch.from_numpy(state).float())
            if len(states) < sequence_length:
                continue  # Skip until we have enough states for a full sequence

            state_sequence = torch.stack(states[-sequence_length:]).unsqueeze(0)  # Shape: (1, sequence_length, state_size)

            # Policy network forward pass
            if random.random() < epsilon:
                # Take a random action
                action = torch.tensor(np.array([env.action_space.sample()]))
                use_policy = False
            else:
                # Otherwise use the policy network to predict the next action
                action_mean, action_std, hidden_state = policy_network(state_sequence, hidden_state)
                normal_dist = torch.distributions.Normal(action_mean, action_std)
                action = normal_dist.sample()
                log_prob = normal_dist.log_prob(action).sum(axis=-1)  # Sum needed if action space is multi-dimensional
                use_policy = True

            next_state, reward, done, _ = env.step(action.item())
            rewards.append(reward)

            if use_policy:
                log_probs.append(log_prob)

            state = next_state

        # Compute the returns for the episode
        returns = compute_returns(normalize_rewards(rewards), gamma)

        # Compute the policy gradients
        policy_gradient = [-log_prob * R for log_prob, R in zip(log_probs, returns)]

        # Update the policy
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_gradient).sum()
        policy_loss.backward()
        optimizer.step()

        print(f"Episode {episode + 1}/{episodes}: Total Reward: {sum(rewards)}")

    print("Training complete")

if __name__ == "__main__":
    env = ElectricCarEnv()
    policy_network = LSTM_PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0])
    train_policy_gradient(env, policy_network)
