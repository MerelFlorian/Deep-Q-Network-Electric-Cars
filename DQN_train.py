import numpy as np
from ElectricCarEnv import ElectricCarEnv
import pandas as pd
from algorithms import DQNAgent
import torch


def train_DQN(env, agent, test_env, test_agent, train_episodes=1, val_episodes=1):
    """
    Function to train the DQN agent.
    """

    total_train_rewards = []
    total_val_rewards = []

    for episode in range(train_episodes):
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

        print(f"Episode {episode + 1}/{train_episodes}: Total Reward: {sum(total_train_rewards) / train_episodes}")
    
    print(f"Training complete for model with lr:{lr}, gamma:{gamma}, activation_fn:{activation_fn}, action_size:{action_size}")
    agent.save("models/DQN/lr:{lr}_gamma:{gamma}_act:{activation_fn}_actsize:{action_size}.pth")

    # Validate the agent
    agent.epsilon = 0  # Set epsilon to 0 to use the learned policy without exploration

    for episode in range(1):
        state = test_env.reset()
        val_episode_rewards = []
        done = False

        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = test_env.step(action)
            val_episode_rewards.append(reward)
            state = next_state

        total_val_rewards.append(sum(val_episode_rewards))

    avg_train_reward = sum(total_train_rewards) / train_episodes
    avg_val_reward = sum(total_val_rewards) / val_episodes
    print(f"Average Training Reward: {avg_train_reward}, Average Validation Reward: {avg_val_reward}")
 
    return avg_val_reward

if __name__ == "__main__":
    train_episodes = 1
    val_episodes = 1

    learning_rates = [0.01, 0.001, 0.0005, 0.0001]
    discount_factors = [0.99, 0.95, 0.90, 0.5]
    activation_functions = [torch.relu, torch.tanh]
    action_sizes = [100, 200, 500, 1000] 
    state_size = 3

    
    best_performance = -float('inf')
    best_params = {}

    for lr in learning_rates:
        for gamma in discount_factors:
            for activation_fn in activation_functions:
                for action_size in action_sizes:

                    # Initialize the environment and agent
                    env = ElectricCarEnv()
                    agent = DQNAgent(state_size, action_size, learning_rate=lr, gamma=gamma, activation_fn=activation_fn)
                    
                    # Initialize the validation environment and agent
                    test_env = ElectricCarEnv()
                    test_agent = DQNAgent(state_size, action_size)
                    test_env.data = pd.read_csv("data/validate_clean.csv")

                    # Train the agent
                    total_reward = train_DQN(env, agent, test_env, test_agent, train_episodes, val_episodes)  

                    # Log and evaluate performance
                    if total_reward > best_performance:
                        best_performance = total_reward
                        best_params = {'lr': lr, 'gamma': gamma, 'activation_fn': activation_fn.__name__, 'action_size': action_size}

    # Print the best parameters
    print(f"Best Parameters: {best_params}")

    agent = DQNAgent(state_size, action_size)
    
    train_DQN(env, agent)
