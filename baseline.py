import numpy as np
import pandas as pd
from ElectricCarEnv import ElectricCarEnv
from algorithms import QLearningAgent, BuyLowSellHigh, EMA
from gym import Env
from typing import Type, Tuple
import sys

# Constants
NUM_EPISODES = 50  # Define the number of episodes for training

def validate_agent(env: Env, agent: Type[QLearningAgent | BuyLowSellHigh | EMA], num_episodes: int, rl = False) -> None:
    """ Function to validate the agent on a validation set.

    Args:
        env (Type[Env]): The environment to validate the agent on.
        agent (Type[QLearningAgent | BuyLowSellHigh | EMA]): The agent to validate.
        num_episodes (int): The number of episodes to validate on.
        rl (bool, optional): Whether the agent is a reinforcement learning agent. Defaults to False.
    """
    # Initialize the total reward
    total_rewards = np.array([])
    # Loop through the episodes
    for episode in range(NUM_EPISODES):
        total_reward = 0
        # Reset the environment
        state = env.reset()
        done = False
        # Loop until the episode is done
        while not done:
            # Choose an action
            action = agent.choose_action(state) if rl else agent.choose_action(env.get_current_price(), state)
            # Take a step
            state, reward, done, _ = env.step(action)
            # Update the total reward
            total_reward += reward
        total_rewards = np.append(total_rewards, total_reward)
    # Compute and return the average reward
    return total_rewards

def qlearning() -> QLearningAgent:
    """ Function to initialize a Q-learning agent.

    Returns:
        QLearningAgent: The Q-learning agent.
    """
    # Initialize the state and action bins
    state_bins = [np.linspace(0, 50, 50), np.arange(0, 25), np.array([0, 1])]  # Discretize battery level, time, availability
    action_bins = np.linspace(-25, 25, 5000)  # Discretize actions (buy/sell amounts)]
    # Create a new agent instance
    test_agent = QLearningAgent(state_bins, action_bins) 
    # Load the Q-table
    test_agent.q_table = np.load('models/best_q_table.npy')

    # Return the agent
    return test_agent

def buylowsellhigh(env: Env) -> BuyLowSellHigh:
    """ Function to initialize a BuyLowSellHigh agent.

    Args:
        env (Env): The environment to initialize the agent on.

    Returns:
        BuyLowSellHigh: The BuyLowSellHigh agent.
    """
    # Create and return a new agent instance
    return BuyLowSellHigh(env.max_battery)

def ema(env: Env) -> EMA:
    """ Function to initialize an EMA agent.

    Args:
        env (Env): The environment to initialize the agent on.

    Returns:
        EMA: The EMA agent.
    """
    # Create and return a new agent instance
    return EMA(3, 12, env.max_battery)

def process_command(env: Env) -> Tuple[QLearningAgent | BuyLowSellHigh | EMA, bool]:
    """ Function to process the command line arguments.

    Args:
        env (Env): The environment to initialize the agent on.

    Returns:
        Tuple[QLearningAgent | BuyLowSellHigh | EMA, bool]: The agent and whether it is a reinforcement learning agent.
    """
    if sys.argv[1] not in ['qlearning', 'blsh', 'ema']:
        print('Invalid command line argument. Please use one of the following: qlearning, blsh, ema')
        exit()
    if sys.argv[1] == 'qlearning':
        return qlearning(), True
    elif sys.argv[1] == 'blsh':
        return buylowsellhigh(env), False
    else:
        return ema(env), False

# Initialize the environment
env = ElectricCarEnv()
# Initialize the agent
test_agent, rl = process_command(env)
# Load validation data into the environment
env.data = pd.read_csv('data/validate_clean.csv') 

# Test the agent
test_performance = validate_agent(env, test_agent, rl)

print(f"Average reward on validation set: {test_performance}")