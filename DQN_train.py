import numpy as np
from ElectricCarEnv import ElectricCarEnv
import pandas as pd
from algorithms import DQNAgent
import torch
import optuna

import optuna

def objective(trial):
    """
    Function to optimize the hyperparameters of the DQN agent.
    """
    # Define the hyperparameter search space using the trial object
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.8, 0.99)
    activation_fn_name = trial.suggest_categorical("activation_fn", ["relu", "tanh"])
    action_size = trial.suggest_categorical("action_size", [100, 200, 500, 1000])
    state_size = 3
    episodes = 100

    activation_fn = torch.relu if activation_fn_name == "relu" else torch.tanh

    # Create the environment and agent
    env = ElectricCarEnv()
    agent = DQNAgent(state_size, action_size, learning_rate=lr, gamma=gamma, activation_fn=activation_fn)

    # Create the validation environment
    test_env = ElectricCarEnv()
    test_env.data = pd.read_csv("data/validate_clean.csv")

    # Train the agent and get validation reward
    validation_reward = train_DQN(env, agent, test_env, episodes, model_save_path=f"models/DQN/lr:{lr}_gamma:{gamma}_act:{activation_fn}_actsize:{action_size}.pth")

    # Optuna aims to maximize the objective
    return validation_reward


def train_DQN(env, agent, test_env, episodes, model_save_path):
    """
    Function to train the DQN agent.
    """

    total_train_rewards = []
    total_val_rewards = []
    state_size = 3

    for episode in range(episodes):
        state = env.reset()
        done = False
        episode_rewards = []

        if not isinstance(state, np.ndarray) or state.shape != (state_size,):
            state = np.reshape(state, (state_size,))  # Ensure the state has the correct shape

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            episode_rewards.append(reward)

            if not isinstance(next_state, np.ndarray) or next_state.shape != (state_size,):
                next_state = np.reshape(next_state, (state_size,)) 
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break

            if len(agent.memory) > agent.batch_size:
                agent.replay()
            
        total_train_rewards.append(sum(episode_rewards))

        print(f"Episode {episode + 1 / episodes}: Total Reward: {sum(total_train_rewards) / episodes}")

    # Validate the agent
    agent.epsilon = 0  # Set epsilon to 0 to use the learned policy without exploration

    for episode in range(episodes):
        state = test_env.reset()
        val_episode_rewards = []
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = test_env.step(action)
            val_episode_rewards.append(reward)
            state = next_state

        total_val_rewards.append(sum(val_episode_rewards))

    avg_train_reward = sum(total_train_rewards) / episodes
    avg_val_reward = sum(total_val_rewards) / episodes
    print(f"Average Training Reward: {avg_train_reward}, Average Validation Reward: {avg_val_reward}")
    agent.save(model_save_path)
 
    return avg_val_reward

if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=1)  

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    
