import numpy as np
from ElectricCarEnv import Electric_Car
from algorithms import QLearningAgent, BuyLowSellHigh, EMA, DQNAgentLSTM, LSTM_PolicyNetwork
from gym import Env
from typing import Type, Tuple
import sys
from collections import defaultdict
from data.data_vis import visualize_bat
from utils import clip
from data.data_vis import plot_revenue

# Constants
NUM_EPISODES = 1 # Define the number of episodes for training

def validate_agent(env: Env, agent: Type[QLearningAgent or BuyLowSellHigh or EMA or DQNAgentLSTM]) -> None:
    """ Function to validate the agent on a validation set.

    Args:
        env (Type[Env]): The environment to validate the agent on.
        agent (Type[QLearningAgent | BuyLowSellHigh | EMA]): The agent to validate.
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
        # Loop until the episode is done
        while not done:
            # Choose an action
            if isinstance(agent, DQNAgentLSTM):
                _, action, _ = agent.choose_action(state)
            elif isinstance(agent, QLearningAgent):
                action = agent.choose_action(state)
            else:
                action = clip(agent.choose_action(state), state)
            
            # Log current state and action if last episode
            if episode == NUM_EPISODES - 1:
                log_env['battery'].append(state[0] * 25)
                log_env['availability'].append(state[7])
                log_env['action'].append(action)
                log_env['price'].append(state[1])
                log_env['date'].append(env.timestamps[env.day - 1])
                log_env['hour'].append(env.hour)                
            # Take a step
            state, reward, done, _, _ = env.step(action)
            # Update the total reward
            total_reward += reward
            # log_env['revenue'].append(total_reward)
        total_rewards = np.append(total_rewards, total_reward)
    # Compute and return the average reward
    return np.mean(total_rewards), log_env

def qlearning() -> QLearningAgent:
    """ Function to initialize a Q-learning agent.

    Returns:
        QLearningAgent: The Q-learning agent.
    """
    # Discretize battery level, time,  price
    state_bins = [
        np.linspace(0, 50, 4), 
        np.array([1, 9, 14, 17, 24]), 
        np.append(np.linspace(0, 100, 20), 2500)
    ]
    
    actions = 17
    mid = int((actions - 1) / 4)

    #  Discretize action bins
    action_bins = np.concatenate((
        np.linspace(-1, 0, mid, endpoint=False), np.linspace(0, 1, actions - 1)
    ))  # Discretize actions (buy/sell amounts)

    # Calculate the size of the Q-table
    qtable_size = [bin.shape[0] for bin in state_bins] + [action_bins.shape[0]]

    # Create a new agent instance
    test_agent = QLearningAgent(state_bins, action_bins, qtable_size, epsilon=0) 
    # Load the Q-table
    test_agent.q_table = np.load('models/Qlearning/room.npy')

    # Return the agent
    return test_agent

def buylowsellhigh() -> BuyLowSellHigh:
    """ Function to initialize a BuyLowSellHigh agent.

    Args:
        env (Env): The environment to initialize the agent on.

    Returns:
        BuyLowSellHigh: The BuyLowSellHigh agent.
    """
    # Create and return a new agent instance
    return BuyLowSellHigh(50)

def ema() -> EMA:
    """ Function to initialize an EMA agent.

    Returns:
        EMA: The EMA agent.
    """
    # Create and return a new agent instance
    return EMA(3, 7, 50)

def process_command(env: Env) -> Tuple[QLearningAgent or BuyLowSellHigh or EMA, str]:
    """ Function to process the command line arguments.

    Args:
        env (Env): The environment to initialize the agent on.

    Returns:
        Tuple[QLearningAgent | BuyLowSellHigh | EMA, str]: The agent and the algorithm name.
    """
    if sys.argv[1] not in ['qlearning', 'blsh', 'ema', "DQN", "all"]:
        print('Invalid command line argument. Please use one of the following: qlearning, blsh, ema')
        exit()
    if sys.argv[1] == 'qlearning':
        return qlearning(), 'Q-learning'
    elif sys.argv[1] == 'blsh':
        return buylowsellhigh(), 'BLSH'
    elif sys.argv[1] == 'ema':
        return ema(), "EMA"
    elif sys.argv[1] =="DQN":
        state_size = 34 
        action_size = 200
        test_agent = DQNAgentLSTM(state_size, action_size)
        test_agent.model = np.load('models/DQN_version_2/lr:0.003083619832717714_gamma:0.29946064465337385_batchsize:168_actsize:200.pth')
        return test_agent, "DQN"
    else: 
        return "all", "All"
        
# Initialize the environment
env = Electric_Car("data/validate.xlsx", "data/f_val.xlsx")

# Initialize the agent
test_agent, algorithm = process_command(env)

# Test the agent
if test_agent == "all":
    # Validate Q-Learning Agent
    q_agent = qlearning()
    q_performance, ql_log_env = validate_agent(env, q_agent)
    print(f"Average reward on validation set for q learning: {q_performance}")

    # Validate BuyLowSellHigh Agent
    blsh_agent = buylowsellhigh(env)
    blsh_performance, blsh_log_env = validate_agent(env, blsh_agent)
    print(f"Average reward on validation set for blsh: {blsh_performance}")

    # Validate EMA Agent
    ema_agent = ema()
    ema_performance, ema_log_env = validate_agent(env, ema_agent)
    print(f"Average reward on validation set for ema: {ema_performance}")

    # Validate DQN Agent
    dqn_agent = DQNAgentLSTM(34, 200)
    dqn_agent.model = np.load('models/DQN_version_2/lr:0.003083619832717714_gamma:0.29946064465337385_batchsize:168_actsize:200.pth')
    dqn_performance, dqn_log_env = validate_agent(env, dqn_agent)
    print(f"Average reward on validation set for dqn: {dqn_performance}")

    # Validate PG Agent
    pg_agent = LSTM_PolicyNetwork(10, 1, 48, 1)
    pg_agent.model = np.load('models/PG.pth')
    pg_performance, pg_log_env = validate_agent(env, pg_agent)
    print(f"Average reward on validation set for pg: {pg_performance}")

    plot_revenue(ql_log_env, blsh_log_env, ema_log_env, dqn_log_env, pg_log_env)

else:
    test_performance, log_env = validate_agent(env, test_agent)
    # Visualize the battery level
    visualize_bat(log_env, algorithm)
    print(f"Average reward on validation set: {test_performance}")