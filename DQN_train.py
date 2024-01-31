import numpy as np
from ElectricCarEnv import Electric_Car
from algorithms import DQNAgentLSTM
import torch
import optuna
import os, csv

def reward_shaping(state, next_state, action, last_price, buy_price, sell_price):
        """Shape the reward such that buying low and selling high is encouraged. 

        Args:
            state (list): The current state of the environment.
            next_state (list): The next state of the environment.
            action (float): The action taken at the current time step.

        Returns:
            float: The shaped reward.
        """
        # Initialize the shaped reward
        shaped_reward = 0

        # Get prices and time from states 
        current_price = state[1]
        current_time = state[2]
        next_price = next_state[1]

        # If action is buying)
        if action > 0:
            # If the agent buys between 3 am and 6 am 
            if 3 <= current_time <= 6:
                shaped_reward += 4
            # If the agent buys again but the price is lower than the previous price
            if current_price < buy_price:
                shaped_reward += 3
            if current_price > sell_price:
                shaped_reward -= 2
            if action > min(action, min(25,  (50 - state[0]) * 0.9)) / 25:
                shaped_reward -= 5
            buy_price = current_price
        # If action is selling
        elif action < 0:
            if current_price >= 2 * buy_price:
                shaped_reward += 4
            if current_price > sell_price:
                shaped_reward += 3
            if current_price < buy_price:
                shaped_reward -= 2
            sell_price = current_price
            if action < max(action, -min(25, state[0] * 0.9)) / 25:
                shaped_reward -= 5
        else:
            if (last_price < current_price < next_price) or (last_price > current_price > next_price):
                shaped_reward += 3
            if (last_price > current_price < next_price) or (last_price < current_price > next_price):
                shaped_reward -= 1
        return shaped_reward, buy_price, sell_price

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
    sequence_length = trial.suggest_int("sequence_length", 1, 30)

    gamma = 0.01
    action_size = 24
    episodes = 20

    # Create the environment and agent
    env = Electric_Car("data/train.xlsx", "data/f_train.xlsx")
    agent = DQNAgentLSTM(len(env.state), action_size, learning_rate=lr, gamma=gamma)

    # Create the validation environment
    test_env = Electric_Car("data/validate.xlsx", "data/f_val.xlsx")
    test_agent = DQNAgentLSTM(len(test_env.state), action_size, learning_rate=lr, gamma=gamma)

    # Train the agent and get validation reward
    validation_reward = train_DQN_LSTM(env, agent, agent.model, test_env, test_agent, episodes, sequence_length, model_save_path=f"models/DQN_version_4/lr:{lr}_gamma:{gamma}_actsize:{action_size}.pth", batch_size=batch_size)

    # Write trial results to CSV
    if not os.path.exists('hyperparameter_tuning_results_DQN_version5.csv'):
        with open('hyperparameter_tuning_results_DQN_version5.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Trial', 'Learning Rate', 'Gamma', 'Action Size', 'Validation Reward'])

    fields = [trial.number, lr, gamma, action_size, validation_reward]
    with open('hyperparameter_tuning_results_DQN_version5.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    # Optuna aims to maximize the objective
    return validation_reward


def train_DQN_LSTM(env, agent, model, test_env, test_agent, episodes, sequence_length, model_save_path, batch_size=1):
    """
    Function to train the DQN agent.
    """

    total_train_rewards, total_val_rewards = [], []
    
    for episode in range(episodes):
        state = env.reset()
        done = False
        states, episode_rewards = [], []
        hidden_state = model.init_hidden(batch_size)
        last_price, buy_price, sell_price = 0, 0, 0
        
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
                shaped_reward, buy_price, sell_price = reward_shaping(state, next_state, action, last_price, buy_price, sell_price)
                agent.remember(state, action, action_index, shaped_reward + reward, next_state, done)

                # Experience replay
                if len(agent.memory) > agent.batch_size:
                    agent.replay()
            last_price = state[1]
            state = next_state 
            states.append(state)
            
            if len(states) < sequence_length:
                continue

            if done:
                break
                
        total_train_rewards.append(sum(episode_rewards))
        validation_reward = sum(validate_agent(test_env, test_agent, states, sequence_length, model, batch_size))
        total_val_rewards.append(validation_reward)

        print(f"Episode {episode + 1}: Train Reward: {sum(episode_rewards)}, Validation Reward: {validation_reward}")
        counter = 0

        # Early stopping
        if episode > 0 and len(total_val_rewards) > 1:
            if total_val_rewards[-1] < total_val_rewards[-2]:
                counter += 1
                if counter == 3:
                        break
                else:
                    counter = 0
    
    # Calculate the rewards
    avg_train_reward = sum(total_train_rewards) / episodes
    avg_val_reward = sum(total_val_rewards) / episodes
    print(f"Average Training Reward: {avg_train_reward}, Average Validation Reward: {avg_val_reward}")
    
    agent.save(model_save_path)
 
    return avg_val_reward

if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=20)  

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    
