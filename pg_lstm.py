import torch
import torch.optim as optim
from gym import Env
from ElectricCarEnv import Electric_Car
import numpy as np
import optuna
import sys
from algorithms import LSTM_PolicyNetwork

def reward_shaping(buys, state, next_state, action, last_price):
        """Shape the reward such that buying low and selling high is encouraged. 

        Args:
            state (list): The current state of the environment.
            next_state (list): The next state of the environment.
            action (float): The action taken at the current time step.

        Returns:
            float: The shaped reward.
        """
        # Initialize the shaped reward
        shaped_reward = 0

        # Get prices, time and features from states 
        current_price = state[1]
        current_time = state[2]
        next_price = next_state[1]
        available = state[7]
        battery_level = state[0]

        buy_price = 0 if len(buys) == 0 else buys.mean()

        # If action is buying)
        if action > 0:
            # Compute the maximum amount of energy that can be bought
            max_buy = min(action, min(25,  (50 - battery_level) * 0.9)) / 25
            # If the agent buys between 3 am and 6 am 
            if 3 <= current_time <= 6:
                shaped_reward += 3
            # If the agent buys at a price less than 30
            if current_price <= 30:
                shaped_reward += 6
            # If the agent buys at a price greater than 70
            if current_price >= 70:
                shaped_reward -= 5
            if battery_level == 50:
                shaped_reward -= 10
            # If the agent buys before 3 am or after 6 am
            if current_time < 3 or current_time > 6:
                shaped_reward -= 5
            # If the agent buys again but the price is 5% higher than the previous price
            if buy_price and current_price > buy_price * 1.05:
                shaped_reward -= 1
            # If the agent buys more than the maximum amount of energy that can be bought
            if action > max_buy / 4.1:
                shaped_reward -= 1
            # If the agent buys between 1/8 and 1/2 of the maximum amount of energy that can be bought
            if action <= max_buy / 8.2:
                shaped_reward += 3
            # Append the buy price (tensor)
            buys = torch.cat((buys, torch.tensor([current_price])))
        # If action is selling
        elif action < 0:
            # Compute the maximum amount of energy that can be sold
            max_sell = max(action, -min(25, battery_level * 0.9)) / 25
            # If the agent sells at a price equal to or greater than the buy price
            if buy_price and current_price >= 2 * buy_price / 0.9:
                shaped_reward += 16
            # If the agent sells at a price greater than 66
            if current_price >= 66 and buy_price:
                shaped_reward += 10
            # If the agent sells at a price less than twice the buy price
            if buy_price and current_price < 2 * buy_price / 0.9:
                shaped_reward -= 16
            if not buy_price:
                shaped_reward -= 5
            # If the agent sells more than the maximum amount of energy that can be sold
            if action < max_sell:
                shaped_reward -= 1
            # Clear the buy history (tensor)
            buys.new_empty(0)
        else:
            # If the agent is unavailable between 9 am and 7 pm
            if 9 <= current_time <= 19 and not available:
                shaped_reward += 5
            # If the price is not a peak or a trough
            if (last_price < current_price < next_price) or (last_price > current_price > next_price):
                shaped_reward += 0.1
            # If the price is a peak or a trough
            if (last_price > current_price < next_price) or (last_price < current_price > next_price):
                shaped_reward -= 0.2
        return shaped_reward, buys
    
def compute_returns(rewards: list, gamma=0.99) -> torch.Tensor:
    """ Computes the discounted returns for a list of rewards.

    Args:
        rewards (list): The list of rewards.
        gamma (float, optional): The discount factor. Defaults to 0.99.

    Returns:
        list: The discounted returns.
    """
    R = 0
    returns = []
    for r in reversed(rewards):
        R = r + gamma * R
        returns.insert(0, R)
    return torch.tensor(returns)

def objective(trial: optuna.Trial) -> float:
    """ The objective function for the Optuna hyperparameter optimization.

    Args:
        env (Env): The environment to train on.
        trial (optuna.Trial): The trial object.

    Returns:
        float: The evaluation metric.
    """
    # Define the hyperparameters using the trial object
    lr = trial.suggest_float('lr', 3e-4, 1e-2, log=True)
    gamma = 0
    noise = trial.suggest_float('noise_std', 2, 15)
    noise_decay = trial.suggest_float('noise_decay', 0.7, 0.95)
    hidden_size = trial.suggest_categorical('hidden_size', [32, 48, 64])
    clipping = trial.suggest_int('clipping', 1, 10)
    sequence_length = trial.suggest_int('sequence_length', 1, 30)

    # Create a new model with these hyperparameters
    policy_network = LSTM_PolicyNetwork(len(env.state), 1, hidden_size, 1).to("mps")
    # Train the model and return the evaluation metric
    total_reward = train_policy_gradient(env, val_env, policy_network, lr=lr, gamma=gamma, noise_std=noise, noise_decay=noise_decay, sequence_length=sequence_length, clipping=clipping, name=f"pg_{trial.number}.pth")
    return -total_reward


def train_policy_gradient(env: Env, val_env: Env, policy_network: LSTM_PolicyNetwork, episodes = 20, lr = 0.0007, gamma = 0, 
                          noise_std = 0.1, noise_decay = 0.9, sequence_length=7, clipping = 4, save = True, name = "pg.pth"):
    """ Trains a policy network using the policy gradient algorithm.

    Args:
        env (Env): The environment to train on.
        val_env (Env): The environment to validate on.
        policy_network (PolicyNetwork): The policy network to train.
        episodes (int, optional): The number of episodes to train for. Defaults to 10.
        lr (float, optional): The learning rate. Defaults to 0.0007.
        gamma (float, optional): The discount factor. Defaults to 0.
        noise_std (float, optional): The standard deviation of the Gaussian noise to add to the actions. Defaults to 0.1.
        noise_decay (float, optional): The decay rate of the noise. Defaults to 0.99.
        sequence_length (int, optional): The length of the sequence of states to use. Defaults to 7.
        clipping (int, optional): The gradient clipping value. Defaults to 1.
        save (bool, optional): Whether to save the model. Defaults to False.
    """
    # Initialize the optimizer
    optimizer = optim.Adam(policy_network.parameters(), lr=lr, weight_decay=0.0001)
    batch_size = 1  # Assuming we're dealing with single episodes

    device = torch.device("mps")

    # Keep track of the validation rewards
    v_rewards = []
    counter = 0
    buys, last_price = torch.from_numpy(np.array([])), 0

    for episode in range(episodes):
        # Reset the environment
        state = env.reset()
        # Initialize the lists to store the states, rewards and log probabilities
        states, rewards, total_reward, log_probs = [], [], 0, []
        hidden_state = policy_network.init_hidden(device, batch_size)
        done = False

        while not done:
            # Prepare the sequence of states
            if len(states) < sequence_length:
                next_state, reward, done, _, _ = env.step(0)
                next_state = torch.from_numpy(next_state).float().to(device)
            else:
                state_sequence = torch.stack(states[-sequence_length:]).unsqueeze(0)
                # Policy network forward pass
                action_mean, action_std, hidden_state = policy_network(state_sequence, hidden_state)
                normal_dist = torch.distributions.Normal(action_mean, action_std)
                # Sample an action from the normal distribution
                sampled_action = normal_dist.sample()
                # Generate noise to encourage exploration
                noise = torch.from_numpy(np.random.normal(0, noise_std, size=sampled_action.shape).astype(np.float32)).to(device)
                # Add the noise to the action
                action = sampled_action + noise
                log_prob = normal_dist.log_prob(action).sum(axis=-1)
                # Take a step in the environment
                next_state, reward, done, _, _ = env.step(action.item())
                next_state = torch.from_numpy(next_state).float().to(device)
                # Store the reward and log probability
                shaped_reward, buys = reward_shaping(buys, state, next_state, action, last_price)
                rewards.append(shaped_reward + reward)
                total_reward += reward

                log_probs.append(log_prob)
            last_price = state[1]
            state = next_state
            states.append(state)
            if len(states) < sequence_length:
                continue

        # Compute the returns for the episode
        returns = compute_returns(rewards, gamma).to(device, dtype=torch.float32)

        # Compute the policy gradients
        policy_gradient = [-log_prob * R for log_prob, R in zip(log_probs, returns)]

        # Update the policy
        optimizer.zero_grad()
        policy_loss = torch.stack(policy_gradient).sum()
        policy_loss.backward()

        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(policy_network.parameters(), clipping)

        optimizer.step()

        # Decay the noise
        noise_std *= noise_decay

        print(f"Episode {episode + 1}/{episodes}: Total training reward: {total_reward}") 

        states = []

        # Validate the model
        with torch.no_grad():
            state = val_env.reset()
            done = False
            total_reward = 0
            while not done:
                if isinstance(state, np.ndarray):
                    state = torch.from_numpy(state).float()
                else:  # If 'state' is already a tensor, just ensure it's the right type
                    state = state.float()
                state = state.to(device)
                states.append(state)
                if len(states) < sequence_length:
                    next_state, reward, done, _, _ = val_env.step(0)
                else:
                    state_sequence = torch.stack(states[-sequence_length:]).unsqueeze(0)
                    action_mean, action_std, hidden_state = policy_network(state_sequence, hidden_state)
                    normal_dist = torch.distributions.Normal(action_mean, action_std)
                    action = normal_dist.sample()
                    next_state, reward, done, _, _ = val_env.step(action.item())
                    total_reward += reward
                
                state = next_state
                if len(states) < sequence_length:
                    continue

            print(f"Validation reward: {total_reward}")
            # Keep track of the validation rewards for early stopping
            v_rewards.append(total_reward)
            if episode == 0:
                best_reward = total_reward
            else:
                # Early stopping
                if v_rewards[-1] < v_rewards[-2]:
                    counter += 1
                    if counter == 4:
                        return best_reward
                else:
                    counter = 0
                # Save the model if it's the best one so far
                if total_reward > best_reward:
                    best_reward = total_reward
                    if save:
                        torch.save(policy_network.state_dict(), f"models/{name}")

    print("Training complete")
    return best_reward

if __name__ == "__main__":
    # Create the environments
    env = Electric_Car("data/train.xlsx", "data/f_train.xlsx")
    val_env = Electric_Car("data/validate.xlsx", "data/f_val.xlsx")
    if len(sys.argv) == 2:
        if sys.argv[1] == 'tune':
            study = optuna.create_study()  # Create a study object
            study.optimize(objective, n_trials=20)  # Optimize the objective over 50 trials

            print("Best trial:")
            trial = study.best_trial
            print(f"  Value: {-trial.value}")
            print("  Params: ")
            for key, value in trial.params.items():
                print(f"    {key}: {value}")
        elif sys.argv[1] == 'train':
            # Create a new model with the best hyperparameters
            policy_network = LSTM_PolicyNetwork(10, 1, 48, 1).to("mps")
            # Load the best model weights
            train_policy_gradient(env, val_env, policy_network, episodes=80, lr=0.0025364074637955723, gamma=0, noise_std = 10.555213987532536, noise_decay=0.8789787544106076, clipping=4, sequence_length=3, save=True)
        else:
            print('Invalid command line argument. Please use one of the following: tune, train')
            exit()
    else:
        print('Missing line argument. Correct usage: python pg_train.py <tune/train>')
        exit()
