import torch.nn as nn
import torch
import torch.optim as optim
from gym import Env
from ElectricCarEnv import Electric_Car
import random
import numpy as np
import optuna
import pandas as pd
import sys

class LSTM_PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, lstm_layers=1):
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

def objective(trial: optuna.Trial) -> float:
    """ The objective function for the Optuna hyperparameter optimization.

    Args:
        env (Env): The environment to train on.
        trial (optuna.Trial): The trial object.

    Returns:
        float: The evaluation metric.
    """
    # Define the hyperparameters using the trial object
    lr = trial.suggest_loguniform('lr', 1e-5, 9e-3)
    gamma = trial.suggest_float('gamma', 0.01, 0.6)
    noise = trial.suggest_float('noise_std', 0.01, 25)
    noise_decay = trial.suggest_float('noise_decay', 0.8, 1)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 48, 64, 128])
    clipping = trial.suggest_int('clipping', 1, 10)
    layers = trial.suggest_categorical('lstm_layers', [2, 3])
    sequence_length = trial.suggest_int('sequence_length', 3, 48)

    # Create a new model with these hyperparameters
    policy_network = LSTM_PolicyNetwork(env.observation_space.shape[0], env.action_space.shape[0], hidden_size, layers)

    # Train the model and return the evaluation metric
    total_reward = train_policy_gradient(env, policy_network, lr=lr, gamma=gamma, noise_std=noise, noise_decay=noise_decay, sequence_length=sequence_length, clipping=clipping)
    return -total_reward


def train_policy_gradient(env: Env, val_env: Env, policy_network: LSTM_PolicyNetwork, episodes = 7, lr = 0.0007, gamma = 0, 
                          noise_std = 0.1, noise_decay = 0.9, sequence_length=7, clipping = 4, save = False):
    """ Trains a policy network using the policy gradient algorithm.

    Args:
        env (Env): The environment to train on.
        val_env (Env): The environment to validate on.
        policy_network (PolicyNetwork): The policy network to train.
        episodes (int, optional): The number of episodes to train for. Defaults to 10.
        lr (float, optional): The learning rate. Defaults to 0.0007.
        gamma (float, optional): The discount factor. Defaults to 0.
        noise_std (float, optional): The standard deviation of the Gaussian noise to add to the actions. Defaults to 0.1.
        noise_decay (float, optional): The decay rate of the noise. Defaults to 0.99.
        sequence_length (int, optional): The length of the sequence of states to use. Defaults to 7.
        clipping (int, optional): The gradient clipping value. Defaults to 1.
        save (bool, optional): Whether to save the model. Defaults to False.
    """
    # Initialize the optimizer
    optimizer = optim.Adam(policy_network.parameters(), lr=lr, weight_decay=0.0001)
    batch_size = 1  # Assuming we're dealing with single episodes

    for episode in range(episodes):
        # Reset the environment
        state = env.reset()
        # Initialize the lists to store the states, rewards and log probabilities
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
            # Otherwise use the policy network to predict the next action
            action_mean, action_std, hidden_state = policy_network(state_sequence, hidden_state)
            normal_dist = torch.distributions.Normal(action_mean, action_std)
            # Sample an action from the normal distribution
            sampled_action = normal_dist.sample()
            # Generate noise to encourage exploration
            noise = np.random.normal(0, noise_std, size=sampled_action.shape)
            # Add the noise to the action
            action = sampled_action + noise
            log_prob = normal_dist.log_prob(action).sum(axis=-1)  # Sum needed if action space is multi-dimensional

            next_state, reward, done, _, _ = env.step(action.item())
            rewards.append(reward)
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

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(policy_network.parameters(), clipping)

        optimizer.step()

        # Decay the noise
        noise_std *= noise_decay

        print(f"Episode {episode + 1}/{episodes}: Total training reward: {sum(rewards)}") 

        # Validate the model
        with torch.no_grad():
            state = val_env.reset()
            done = False
            total_reward = 0
            while not done:
                state = torch.from_numpy(state).float().unsqueeze(0).unsqueeze(0)
                action_mean, action_std, hidden_state = policy_network(state, hidden_state)
                normal_dist = torch.distributions.Normal(action_mean, action_std)
                action = normal_dist.sample()
                next_state, reward, done, _, _ = val_env.step(action.item())
                total_reward += reward
                state = next_state

            print(f"Validation reward: {total_reward}")

            if save:
                if episode == 0:
                    best_reward = total_reward
                else:
                    if total_reward > best_reward:
                        best_reward = total_reward
                        if best_reward > 0:
                            torch.save(policy_network.state_dict(), "models/pg.pth")

    print("Training complete")
    return best_reward

if __name__ == "__main__":
    # Create the environments
    env = Electric_Car("data/train.xlsx")
    val_env = Electric_Car("data/validate.xlsx")

    if sys.argv[1] == 'tune':
        study = optuna.create_study()  # Create a study object
        study.optimize(objective, n_trials=50)  # Optimize the objective over 50 trials

        print("Best trial:")
        trial = study.best_trial
        print(f"  Value: {-trial.value}")
        print("  Params: ")
        for key, value in trial.params.items():
            print(f"    {key}: {value}")
    elif sys.argv[1] == 'train':
        # Create a new model with the best hyperparameters
        policy_network = LSTM_PolicyNetwork(8, 1, 48, 3)
        # Load the best model weights
        train_policy_gradient(env, val_env, policy_network, episodes=5, lr=0.005, gamma=0.37, noise_std = 0.5, clipping=3, sequence_length=21, save=True)
    else:
        print('Invalid command line argument. Please use one of the following: tune, train')
        exit()
