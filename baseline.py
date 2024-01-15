import numpy as np
import pandas as pd
from ElectricCarEnv import ElectricCarEnv
from algorithms import QLearningAgent, BuyLowSellHigh, EMA
from gym import Env
from typing import Type
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
    total_rewards = 0
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
        total_rewards += reward
    # Compute and return the average reward
    return total_rewards / num_episodes

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
    return EMA(env.max_battery)

# Initialize the environment
env = ElectricCarEnv()
# Initialize the agent
test_agent = qlearning() if sys.argv[1] == 'qlearning' else buylowsellhigh(env) if sys.argv[1] == 'blsh' else ema(env)
# Load validation data into the environment
env.data = pd.read_csv('data/validate_clean.csv') 

# Test the agent
test_performance = validate_agent(env, test_agent, NUM_EPISODES)

print(f"Average reward on validation set: {test_performance}")