import numpy as np
import pandas as pd
from ElectricCarEnv import ElectricCarEnv
from algorithms import QLearningAgent, BuyLowSellHigh, EMA
from gym import Env
from typing import Type, Tuple
import sys
from datetime import datetime
from collections import defaultdict
from data.data_vis import visualize_bat

# Constants
NUM_EPISODES = 50  # Define the number of episodes for training

def validate_agent(env: Env, agent: Type[QLearningAgent or BuyLowSellHigh or EMA], rl = False) -> None:
    """ Function to validate the agent on a validation set.

    Args:
        env (Type[Env]): The environment to validate the agent on.
        agent (Type[QLearningAgent | BuyLowSellHigh | EMA]): The agent to validate.
        rl (bool, optional): Whether the agent is a reinforcement learning agent. Defaults to False.
    """
    # Initialize the total reward
    total_rewards = np.array([])
    # Loop through the episodes
    for _ in range(NUM_EPISODES):
        total_reward = 0
        # Reset the environment
        state = env.reset()
        done = False
        log_env = defaultdict(list)
        # Loop until the episode is done
        while not done:
            # Choose an action
            action = agent.choose_action(state) if rl else agent.choose_action(env.get_current_price(), state)
            # Log current state and action if last episode
            if episode == NUM_EPISODES - 1:
                log_env['battery'].append(state[0])
                log_env['availability'].append(state[2])
                log_env['action'].append(action)
                log_env['price'].append(env.get_current_price())
            # Take a step
            state, reward, done, _ = env.step(action)
            # Get datetime
            date = datetime.strptime(f"{env.data.iloc[_['step']]['date']} {0 if state[1] == 24 else state[1]:02d}:00:00", "%Y-%m-%d %H:%M:%S")
            #Log date if last episode
            if episode == NUM_EPISODES - 1:
                log_env['date'].append(date)
            # Update the total reward
            total_reward += reward
        total_rewards = np.append(total_rewards, total_reward)
    # Compute and return the average reward
    return np.mean(total_rewards), log_env

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

def process_command(env: Env) -> Tuple[QLearningAgent or BuyLowSellHigh or EMA, bool]:
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
test_performance, log_env = validate_agent(env, test_agent, rl)

# Visualize the battery level
visualize_bat(log_env)


print(f"Average reward on validation set: {test_performance}")