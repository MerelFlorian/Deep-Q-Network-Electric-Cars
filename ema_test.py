import pandas as pd
from ElectricCarEnv import ElectricCarEnv
from algorithms import EMA

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
            #print(reward)
            total_rewards += reward
    
    average_reward = total_rewards / num_episodes
    return average_reward

# Environment and Agent Initialization
env = ElectricCarEnv()
agent = EMA(9, 21, 2, 2, env.max_battery)

# Load validation data into the environment
env.data = pd.read_csv('data/validate_clean.csv') 

# Test the agent
num_test_episodes = 10  # Define the number of episodes for testing
test_performance = validate_agent(env, agent, num_test_episodes)

print(f"Average reward on validation set: {test_performance}")