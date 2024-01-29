import numpy as np
from ElectricCarEnv import Electric_Car
import pandas as pd
from algorithms import DQNAgent
import torch
import optuna
import os, csv


def objective(trial):
    """
    Function to optimize the hyperparameters of the DQN agent.
    """
    # Define the hyperparameter search space using the trial object
    lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
    gamma = trial.suggest_float("gamma", 0.1, 0.3)
    action_size = trial.suggest_categorical("action_size", [100, 200, 500])
    #batch_size = trial.suggest_categorical("batch_size", [24, 48, 168]) # 1 day, 2 days, 1 week
    #sequence_length = trial.suggest_categorical("sequence_length", [7, 24, 48]) # 1 week, 1 day, 2 days
    episodes = 30

    # Create the environment and agent
    env = Electric_Car("data/train.xlsx", "data/f_train.xlsx")
    agent = DQNAgent(len(env.state), action_size, learning_rate=lr, gamma=gamma)

    # Create the validation environment
    test_env = Electric_Car("data/validate.xlsx", "data/f_val.xlsx")

    # Train the agent and get validation reward
    validation_reward = train_DQN(env, agent, agent.model, test_env, episodes, model_save_path=f"models/DQN_version_4/lr:{lr}_gamma:{gamma}_actsize:{action_size}.pth")

    # Write trial results to CSV
    if not os.path.exists('hyperparameter_tuning_results_DQN_version4.csv'):
        with open('hyperparameter_tuning_results_version4.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Trial', 'Learning Rate', 'Gamma', 'Action Size', 'Validation Reward'])

    fields = [trial.number, lr, gamma, action_size, validation_reward]
    with open('hyperparameter_tuning_results_DQN_version4.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    # Optuna aims to maximize the objective
    return validation_reward


# def train_DQN(env, agent, model, test_env, episodes, sequence_length, model_save_path, batch_size=1):
#     """
#     Function to train the DQN agent.
#     """

#     total_train_rewards, total_val_rewards = [], []
    
#     for episode in range(episodes):
#         state = env.reset()
#         done = False
#         states, episode_rewards = [], []
#         hidden_state = model.init_hidden(batch_size)

#         if not isinstance(state, np.ndarray) or state.shape != (len(env.state),):
#             state = np.reshape(state, (len(env.state),))  # Ensure the state has the correct shape
        
#         while not done:
#             # Prepare the sequence of states
#             if len(states) < sequence_length:
#                 next_state, reward, done, _, _ = env.step(0)
#             else:
#                 state_sequence = torch.stack(states[-sequence_length:]).unsqueeze(0)  # Shape: (1, sequence_length, state_size)
#                 action_index, action, hidden_state = agent.choose_action(state_sequence, hidden_state)

#                 next_state, reward, done, _, _ = env.step(action)
#                 episode_rewards.append(reward)
#                 agent.remember(state, action, action_index, reward, next_state, done)

#                 if len(agent.memory) > agent.batch_size:
#                     agent.replay()
              
#             state = next_state
#             states.append(torch.from_numpy(state).float())
            
#             if len(states) < sequence_length:
#                 continue

#             if done:
#                 break
                
#         total_train_rewards.append(sum(episode_rewards))

#         print(f"Episode {episode + 1}: Total Reward: {sum(episode_rewards)}")

#     # Validate the agent
#     agent.epsilon = 0  # Set epsilon to 0 to use the learned policy without exploration
#     counter = 0

#     for episode in range(episodes):
#         state = test_env.reset()
#         val_episode_rewards, states = [], []
#         done = False
#         hidden_state = model.init_hidden(batch_size)

#         if not isinstance(state, np.ndarray) or state.shape != (len(test_env.state),):
#             state = np.reshape(state, (len(test_env.state),))  # Ensure the state has the correct shape

#         while not done:
#             if len(states) < sequence_length:
#                 states.append(torch.from_numpy(state).float())
#                 continue
        
#             state_sequence = torch.stack(states[-sequence_length:]).unsqueeze(0)  # Shape: (1, sequence_length, state_size)
#             action_index, action, hidden_state = agent.choose_action(state_sequence, hidden_state)
#             next_state, reward, done, _, _ = test_env.step(action)
#             val_episode_rewards.append(reward)

#             state = next_state
#             states.append(torch.from_numpy(state).float())

#         total_val_rewards.append(sum(val_episode_rewards))

#         # Early stopping
#         if episode > 0 and len(total_val_rewards) > 1:
#             if total_val_rewards[-1] < total_val_rewards[-2]:
#                 counter += 1
#                 if counter == 3:
#                         break
#                 else:
#                     counter = 0

#     avg_train_reward = sum(total_train_rewards) / episodes
#     avg_val_reward = sum(total_val_rewards) / episodes
#     print(f"Average Training Reward: {avg_train_reward}, Average Validation Reward: {avg_val_reward}")
    
#     agent.save(model_save_path)
 
#     return avg_val_reward

def train_DQN(env, agent, model, test_env, episodes, model_save_path):
    """
    Function to train the DQN agent.
    """

    total_train_rewards, total_val_rewards = [], []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        states, episode_rewards = [], []

        if not isinstance(state, np.ndarray) or state.shape != (len(env.state),):
            state = np.reshape(state, (len(env.state),))  # Ensure the state has the correct shape
        
        while not done:    
            action_index, action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)
            episode_rewards.append(reward)
            agent.remember(state, action, action_index, reward, next_state, done)

            if len(agent.memory) > agent.replay_size:
                agent.replay()
            
            state = next_state
            states.append(torch.from_numpy(state).float())
                
        total_train_rewards.append(sum(episode_rewards))

        print(f"Episode {episode + 1}: Total Reward: {sum(episode_rewards)}")

    # Validate the agent
    agent.epsilon = 0  # Set epsilon to 0 to use the learned policy without exploration
    counter = 0

    for episode in range(episodes):
        state = test_env.reset()
        val_episode_rewards, states = [], []
        done = False

        if not isinstance(state, np.ndarray) or state.shape != (len(test_env.state),):
            state = np.reshape(state, (len(test_env.state),))  # Ensure the state has the correct shape

        while not done:
            action_index, action  = agent.choose_action(state)
            next_state, reward, done, _, _ = test_env.step(action)
            val_episode_rewards.append(reward)
            state = next_state
            states.append(torch.from_numpy(state).float())
           
        total_val_rewards.append(sum(val_episode_rewards))

        # Early stopping
        if episode > 0 and len(total_val_rewards) > 1:
            if total_val_rewards[-1] < total_val_rewards[-2]:
                counter += 1
                if counter == 3:
                        break
                else:
                    counter = 0

    avg_train_reward = sum(total_train_rewards) / episodes
    avg_val_reward = sum(total_val_rewards) / episodes
    print(f"Average Training Reward: {avg_train_reward}, Average Validation Reward: {avg_val_reward}")
    
    agent.save(model_save_path)
    return avg_val_reward

if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    
