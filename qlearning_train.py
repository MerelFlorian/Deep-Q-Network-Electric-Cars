import numpy as np
from ElectricCarEnv import Electric_Car
from algorithms import QLearningAgent
import matplotlib.pyplot as plt
import pandas as pd
from utils import save_best_q
import optuna
import os, csv

def objective(trial):
    # Suggest values for the hyperparameters
    lr = trial.suggest_float('lr', 1e-5, 1e-1, log=True)
    gamma = trial.suggest_float('gamma', 0.85, 0.99)
    shape_weight = trial.suggest_float('shape_weight', 0.0, 1.0)
    epsilon_decay = trial.suggest_float('epsilon_decay', 0.8, 1.0)

    num_episodes = 100

    # Discretize battery level, time,  price
    state_bins = [
    np.linspace(0, 50, 4), 
    np.arange(0, 25, 4), 
    np.concatenate([
        np.linspace(0, 100, 15),  
        np.linspace(100, 2500, 2) 
    ])  # Bins for price
    ]

    #  Discretize action bins
    action_bins = np.linspace(-1, 1, 50)  # Discretize actions (buy/sell amounts)

    # Calculate the size of the Q-table
    qtable_size = [bin.shape[0] for bin in state_bins] + [action_bins.shape[0]]

    agent = QLearningAgent(state_bins, action_bins, qtable_size, learning_rate=lr, discount_factor=gamma, epsilon_decay=epsilon_decay, shape_weight=shape_weight)
    # Environment and Agent Initialization
    env = Electric_Car("./data/train.xlsx", "./data/f_train.xlsx")

    # Load validation data into the environment
    test_env = Electric_Car("data/validate.xlsx", "data/f_val.xlsx")
    test_agent = QLearningAgent(state_bins, action_bins, qtable_size, learning_rate=lr, discount_factor=gamma, epsilon_decay=epsilon_decay, shape_weight=shape_weight) 

    validation_reward = train_qlearning(env, agent, num_episodes, test_env, test_agent, model_save_path=f"models/Qlearning/lr:{lr}_gamma:{gamma}_shape_weight:{shape_weight}_epsilon_decay:{epsilon_decay}.pth")

    # Write trial results to CSV
    if not os.path.exists('hyperparameter_tuning_results_qlearning.csv'):
        with open('hyperparameter_tuning_results_qlearning.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Trial', 'Learning Rate', 'Gamma', 'Shape Weight', 'Epsilon Decay','Validation Reward'])

    fields = [trial.number, lr, gamma, shape_weight, epsilon_decay, validation_reward]
    with open('hyperparameter_tuning_results_qlearning.csv', 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(fields)

    # Optuna aims to maximize the objective
    return validation_reward
 

def validate_agent(test_env, test_agent, qtable):
    # Initialize the total reward
    total_reward = 0
    # Reset the environment
    state = test_env.reset()
    done = False
    test_agent.q_table = qtable
    # Loop until the episode is done
    while not done:
        # Choose an action
        action = test_agent.choose_action(state)
        # Take a step
        state, reward, done, _,_ =  test_env.step(action)
        # Update the total reward
        total_reward += reward
    
    return total_reward
  
def train_qlearning(env, agent, num_episodes, test_env, test_agent, model_save_path):
    """
    Function to train a Q-learning agent.
    """
    # Initialize the total reward and validation reward
    total_rewards = []
    highest_reward = -np.inf

    # Define early stopping criteria
    patience = 5
    early_stopping_counter = 0

    # Loop through the episodes
    for episode in range(num_episodes):
        # Reset the environment
        state = env.reset()
        done = False
        total_reward = 0
        battery_levels = []
        agent.update_epsilon()
        last_price = 0

        # Loop until the episode is done
        while not done:
            # Choose an action and take a step
            action = agent.choose_action(state)
            next_state, reward, done, _, _ = env.step(action)

            # Update the agent if not in the last episode
            if not done:
                agent.update(state, action, reward, next_state, last_price)
                battery_levels.append(env.battery_level)
            
            # Update the total reward
            last_price= state[1]
            state = next_state
            total_reward += reward

        # Keep track of the total reward for this episode
        total_rewards.append(total_reward)

        # Run validation 
        validation_reward = validate_agent(test_env, test_agent, qtable=agent.q_table)
        
        # Check and update the highest reward and best episode
        if validation_reward > highest_reward:
            highest_reward = validation_reward
            best_episode = episode
            best_q_table = agent.q_table.copy()
            early_stopping_counter = 0  # Reset early stopping counter
        else:
            early_stopping_counter += 1
            print(f"Early stopping counter: {early_stopping_counter}")

        # Check for early stopping
        if early_stopping_counter >= patience:
            print(f"Early stopping at episode {episode} due to lack of improvement in validation reward.")
            save_best_q(best_q_table, highest_reward, best_episode, model_save_path)
            break

        # Print the total reward for this episode
        print(f"Episode {episode} reward: {total_reward} | Validation reward: {validation_reward} | epsilon: {agent.epsilon} | 0s in Q-table: {np.sum(agent.q_table == 0) / agent.q_table.size}")
        
    # Save the best Q-table
    save_best_q(best_q_table, highest_reward, best_episode, model_save_path)
    
    return highest_reward

if __name__ == "__main__":

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=50)  

    print("Best trial:")
    trial = study.best_trial

    print(f"Value: {trial.value}")
    print("Params: ")
    for key, value in trial.params.items():
        print(f"{key}: {value}")

    
