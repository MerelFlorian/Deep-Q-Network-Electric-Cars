import numpy as np
import pandas as pd
from ElectricCarEnv import Electric_Car
from algorithms import DQNAgent

def validate_agent(env, agent, num_episodes=1):
    """
    Function to validate the agent on a validation set.
    """
    episode_rewards = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)

            total_reward += reward
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        print(f"Validation Episode {episode + 1}/{num_episodes}, Total Reward: {total_reward}")

    average_reward = sum(episode_rewards) / num_episodes
    print(f"Average Validation Reward over {num_episodes} Episodes: {average_reward}")
    return episode_rewards


def train_DQN(env, agent):
    """
    Function to train the DQN agent.
    """
    episodes = 1
    rewards = []

    for episode in range(episodes):
        state = env.reset()
        done = False

        if not isinstance(state, np.ndarray) or state.shape != (state_size,):
            state = np.reshape(state, (state_size,))  # Ensure the state has the correct shape

        while not done:
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            rewards.append(reward)

            if not isinstance(next_state, np.ndarray) or next_state.shape != (state_size,):
                next_state = np.reshape(next_state, (state_size,)) 
            
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                break
            if len(agent.memory) > agent.batch_size:
                agent.replay()

        print(f"Episode {episode + 1}/{episodes}: Total Reward: {sum(rewards)}")

    print("Training complete")

    agent.save("models/dqn_model.pth")

if __name__ == "__main__":
    env = ElectricCarEnv()
    state_size = 3  
    action_size = 5000  
    agent = DQNAgent(state_size, action_size)
    
    # train_DQN(env, agent)

    test_env = ElectricCarEnv()
    test_env.data = pd.read_csv('data/validate_clean.csv')
    test_agent = DQNAgent(state_size, action_size)
    test_agent.model = np.load('models/dqn_model.pth')
    validate_agent(test_env, test_agent, True)