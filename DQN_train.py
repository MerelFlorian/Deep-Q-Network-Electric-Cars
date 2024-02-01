import numpy as np
from ElectricCarEnv import Electric_Car
from algorithms import DQNAgentLSTM
import torch
import optuna
import os, csv
from copy import deepcopy


def validate_agent(test_env, test_agent, states, sequence_length, model, batch_size):
    """
    Funtion to validate the agent on a validation set.
    """
    test_agent.epsilon = 0  # Set epsilon to 0 to use the learned policy without exploration
    state = test_env.reset()
    val_rewards, states = [], []
    done = False
    hidden_state = model.init_hidden(batch_size)

    # Ensure the state has the correct shape
    if not isinstance(state, np.ndarray) or state.shape != (len(test_env.state),):
        state = np.reshape(state, (len(test_env.state),)) 

    while not done:
        if len(states) < sequence_length:
            next_state, reward, done, _, _ = test_env.step(0)
        else:
            state_sequence = torch.stack(states[-sequence_length:]).unsqueeze(0)  # Shape: (1, sequence_length, state_size)
            action_index, action, hidden_state = test_agent.choose_action(state_sequence, hidden_state)
            next_state, reward, done, _, _ = test_env.step(action)
        
        val_rewards.append(reward)
        state = torch.from_numpy(next_state).float().to(test_agent.device)
        states.append(state)

        if len(states) < sequence_length:
            continue 

    return val_rewards


def objective(trial):
    """
    Function to optimize the hyperparameters of the DQN agent.
    """
    # Define the hyperparameter search space using the trial object
    lr = trial.suggest_float("lr", 3e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1, 2, 4, 8, 16, 32, 64, 128])
    sequence_length = trial.suggest_int("sequence_length", 1, 10)

    gamma = 0
    action_size = 10
    episodes = 3

    # Create the environment and agent
    env = Electric_Car("data/train.xlsx", "data/f_train.xlsx")
    agent = DQNAgentLSTM(len(env.state), action_size, learning_rate=lr, gamma=gamma, batch_size=batch_size)

    # Create the validation environment
    test_env = Electric_Car("data/validate.xlsx", "data/f_val.xlsx")
    test_agent = DQNAgentLSTM(len(test_env.state), action_size, learning_rate=lr, gamma=gamma, batch_size=batch_size)

    # Train the agent and get validation reward
    validation_reward = train_DQN_LSTM(env, agent, agent.model, test_env, test_agent, episodes, sequence_length, model_save_path=f"models/DQN_version_4/lr:{lr}_gamma:{gamma}_actsize:{action_size}.pth", batch_size=batch_size)

    # Write trial results to CSV
    if not os.path.exists('hyperparameter_tuning_results_DQN_version5.csv'):
        with open('hyperparameter_tuning_results_DQN_version5.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Trial', 'Learning Rate', 'Gamma', 'Action Size', 'Sequence L', 'Batch', 'Validation Reward'])

    fields = [trial.number, lr, gamma, action_size, sequence_length, batch_size, validation_reward]
    with open('hyperparameter_tuning_results_DQN_version5.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    # Optuna aims to maximize the objective
    return validation_reward


def train_DQN_LSTM(env, agent, model, test_env, test_agent, episodes, sequence_length, model_save_path, batch_size=1):
    """
    Function to train the DQN agent.
    """

    total_train_rewards = []
    prev = -np.inf
    highest_reward = -np.inf

    # Define early stopping criteria
    patience = 3
    early_stopping_counter = 0

    
    for episode in range(episodes):
        state = env.reset()
        done = False
        states, episode_rewards = [], []
        hidden_state = model.init_hidden(batch_size)
        last_price = 0
        
        # Ensure the state has the correct shape
        if not isinstance(state, np.ndarray) or state.shape != (len(env.state),):
            state = np.reshape(state, (len(env.state),))  
        
        while not done:
            # Prepare the sequence of states
            if len(states) < sequence_length:
                next_state, reward, done, _, _ = env.step(0)
                next_state = torch.from_numpy(next_state).float().to(agent.device)
            else:
                state_sequence = torch.stack(states[-sequence_length:]).unsqueeze(0)  # Shape: (1, sequence_length, state_size)
                action_index, action, hidden_state = agent.choose_action(state_sequence, hidden_state)

                next_state, reward, done, _, _ = env.step(action)
                next_state = torch.from_numpy(next_state).float().to(agent.device)
                episode_rewards.append(reward)
                shaped_reward = agent.reward_shaping(state, next_state, action, last_price)
                agent.remember(state, action, action_index, shaped_reward + reward, next_state, done)

                # Experience replay
                if len(agent.memory) > agent.batch_size:
                    agent.replay()

            last_price = state[1]
            state = next_state 
            states.append(state)
            
            if len(states) < sequence_length:
                continue
                
        total_train_rewards.append(sum(episode_rewards))
        validation_reward = sum(validate_agent(test_env, test_agent, states, sequence_length, model, batch_size))

        # Save the best model
        if validation_reward > highest_reward:
            highest_reward = validation_reward
            best_episode = episode
            best_model = deepcopy(agent.model)
        
        # Check and update the highest reward and best episode
        if prev < validation_reward:
            early_stopping_counter = 0  # Reset early stopping counter
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}")

        prev = validation_reward
    
        # Check for early stopping
        if early_stopping_counter >= patience:
            print(f"Early stopping at episode {episode} due to lack of improvement in validation reward.")
            agent.save(best_model, model_save_path)
            break
        
        print(f"Episode {episode + 1}: Train Reward: {sum(episode_rewards)}, Validation Reward: {validation_reward}")

    agent.save(best_model, model_save_path)
 
    return highest_reward

if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    
