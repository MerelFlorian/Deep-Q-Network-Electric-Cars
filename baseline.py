import numpy as np
import pandas as pd
from ElectricCarEnv import ElectricCarEnv
from algorithms import QLearningAgent, BuyLowSellHigh, EMA, DQNAgent
from gym import Env
from typing import Type, Tuple
import sys
from datetime import datetime
from collections import defaultdict
from data.data_vis import visualize_bat, plot_revenue

# Constants
NUM_EPISODES = 20 # Define the number of episodes for training

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
    for episode in range(NUM_EPISODES):
        total_reward = 0
        # Reset the environment
        state = env.reset()
        done = False
        log_env = defaultdict(list)
        # Initialize step to 0
        _ = {'step': 0}
        # Loop until the episode is done
        while not done:
            # Choose an action
            action = agent.choose_action(state) if rl else agent.choose_action(env.get_current_price(), state)
            # Get datetime
            first = state[1] + _['step'] == 0
            time = int(state[1]) if not first else 1
            date = datetime.strptime(f"{env.data.iloc[_['step']]['date']} {time-1:02d}:30:00", "%Y-%m-%d %H:%M:%S")
            # Log current state and action if last episode
            if episode == NUM_EPISODES - 1:
                log_env['battery'].append(state[0])
                log_env['availability'].append(state[2])
                log_env['action'].append(action)
                log_env['price'].append(env.get_current_price())
                log_env['date'].append(date)                
            # Take a step
            state, reward, done, _ = env.step(action)
            # Update the total reward
            total_reward += reward
            log_env['revenue'].append(total_reward)
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
    return EMA(3, 7, env.max_battery)

def process_command(env: Env) -> Tuple[QLearningAgent or BuyLowSellHigh or EMA, bool]:
    """ Function to process the command line arguments.

    Args:
        env (Env): The environment to initialize the agent on.

    Returns:
        Tuple[QLearningAgent | BuyLowSellHigh | EMA, bool]: The agent and whether it is a reinforcement learning agent.
    """
    if sys.argv[1] not in ['qlearning', 'blsh', 'ema', "DQN", "all"]:
        print('Invalid command line argument. Please use one of the following: qlearning, blsh, ema')
        exit()
    if sys.argv[1] == 'qlearning':
        return qlearning(), True, 'Q-learning'
    elif sys.argv[1] == 'blsh':
        return buylowsellhigh(env), False, 'BLSH'
    elif sys.argv[1] == 'ema':
        return ema(env), False, "EMA"
    elif sys.argv[1] =="DQN":
        state_size = 3  
        action_size = 5000 
        test_agent = DQNAgent(state_size, action_size)
        test_agent.model = np.load('models/dqn_model.pth')
        return test_agent, True, "DQN"

    else: 
        return "all", True, "All"
        
# Initialize the environment
env = ElectricCarEnv()
# Initialize the agent
test_agent, rl, algorithm = process_command(env)
# Load validation data into the environment
env.data = pd.read_csv('data/validate_clean.csv') 

# Test the agent
if test_agent == "all":
    # Validate Q-Learning Agent
    q_agent = qlearning()
    q_performance, ql_log_env = validate_agent(env, q_agent, True)
    print(f"Average reward on validation set for q learning: {q_performance}")

    # Validate BuyLowSellHigh Agent
    blsh_agent = buylowsellhigh(env)
    blsh_performance, blsh_log_env = validate_agent(env, blsh_agent)
    print(f"Average reward on validation set for blsh: {blsh_performance}")

    # Validate EMA Agent
    ema_agent = ema(env)
    ema_performance, ema_log_env = validate_agent(env, ema_agent, False)
    print(f"Average reward on validation set for ema: {ema_performance}")

    plot_revenue(ql_log_env, blsh_log_env, ema_log_env)

else:
    test_performance, log_env = validate_agent(env, test_agent, rl)
    # Visualize the battery level
    visualize_bat(log_env, algorithm)
    print(f"Average reward on validation set: {test_performance}")