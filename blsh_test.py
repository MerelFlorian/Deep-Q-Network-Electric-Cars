import pandas as pd
from ElectricCarEnv import ElectricCarEnv
from algorithms import BuyLowSellHigh

def validate_agent(env, agent, num_episodes):
    """"
    Function to validate the agent on a validation set.
    """
    total_rewards = 0
    for episode in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = agent.choose_action(env.get_current_price(), state)
            state, reward, done, _ = env.step(action)
            total_rewards += reward
    
    average_reward = total_rewards / num_episodes
    return average_reward, agent

# Environment and Agent Initialization
env_test = ElectricCarEnv()
agent = BuyLowSellHigh(env_test.max_battery)

# Load validation data into the environment
env_test.data = pd.read_csv('data/validate_clean.csv') 

# Test the agent
num_test_episodes = 50 # Define the number of episodes for testing

test_performance, _ = validate_agent(env_test, agent, num_test_episodes)

print(f"Average reward on validation set: {test_performance}")