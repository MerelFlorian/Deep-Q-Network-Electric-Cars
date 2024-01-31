import numpy as np
from TestEnv import Electric_Car
from gym import Env
from typing import Type, Tuple
import sys
from utils import clip
from agent import QLearningAgent, BuyLowSellHigh, EMA, DQNAgentLSTM, LSTM_PolicyNetwork
from utils import create_features

# Constants
NUM_EPISODES = 1 # Define the number of episodes for validating

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
       
        # Loop until the episode is done
        while not done:
            # Choose an action
            if isinstance(agent, DQNAgentLSTM):
                _, action, _ = agent.choose_action(state)
            elif isinstance(agent, QLearningAgent):
                action = agent.choose_action(state)
            # elif isinstance(agent, LSTM_PolicyNetwork(10, 1, 48, 1)):
            #     action = agent.choose_action(state)
            else:
                action = clip(agent.choose_action(state), state)
             
            # Take a step
            state, reward, done, _, _ = env.step(action)

            # Update the total reward
            total_reward += reward
            
        total_rewards = np.append(total_rewards, total_reward)

    # Compute and return the average reward
    return np.mean(total_rewards)

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
    mid = int((actions - 1) / 2)

    #  Discretize action bins
    action_bins = np.concatenate((
        np.linspace(-1, 0, mid, endpoint=False), np.linspace(0, 1, mid)
    ))  # Discretize actions (buy/sell amounts)

    # Calculate the size of the Q-table
    qtable_size = [bin.shape[0] for bin in state_bins] + [action_bins.shape[0]]

    # Create a new agent instance
    test_agent = QLearningAgent(state_bins, action_bins, qtable_size, epsilon=0) 
    
    # Load the Q-table
    test_agent.q_table = np.load('models/Qlearning.npy')

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


def process_command() -> Tuple[QLearningAgent or BuyLowSellHigh or EMA, str]:
    """ Function to process the command line arguments.

    Args:
        env (Env): The environment to initialize the agent on.

    Returns:
        Tuple[QLearningAgent | BuyLowSellHigh | EMA, str]: The agent and the algorithm name.
    """
    
    # Create data set with added features
    data_path = sys.argv[2]
    create_features(data_path, 'data/f_val.xlsx')

    # Initialize the environment
    env = Electric_Car(data_path, "data/f_val.xlsx")
    
    if sys.argv[1] not in ['qlearning', 'blsh', 'ema', "DQN", "PG"]:
        print('Invalid command line argument. Please use one of the following: qlearning, blsh, ema')
        exit()
    if sys.argv[1] == 'qlearning':
        return env, qlearning(), 'Q-learning'
    elif sys.argv[1] == 'blsh':
        return env, buylowsellhigh(), 'BLSH'
    elif sys.argv[1] == 'ema':
        return env, ema(), "EMA"
    elif sys.argv[1] =="DQN":
        state_size = 34 
        action_size = 200
        test_agent = DQNAgentLSTM(state_size, action_size)
        test_agent.model = np.load('models/DQN.pth')
        return env, test_agent, "DQN"
    elif sys.argv[1] == "PG":
        test_agent = LSTM_PolicyNetwork(10, 1, 48, 1)
        test_agent.model = np.load('models/PG.pth')
        return env, test_agent, 'PG'
    else: 
        print('Invalid command line argument. Please use one of the following: qlearning, blsh, ema')
        exit()

if __name__ == "__main__":
    
    # Initialize the agent
    env, test_agent, algorithm = process_command()
    
    # Validate the agent
    test_performance = validate_agent(env, test_agent)
    print(f"Average reward on validation set: {test_performance}")
        