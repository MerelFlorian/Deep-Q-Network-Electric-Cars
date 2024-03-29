import numpy as np
import random
from typing import Type, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
from collections import deque

class QLearningAgent:
    """
    Implements a simple tabular Q-learning agent for the electric car trading problem.
    """	
    def __init__(self, state_bins, action_bins, qtable_size, learning_rate=0.000001, discount_factor=0, epsilon=1, epsilon_decay=0.95, min_epsilon=0, max_battery=50, shape_weight = 1):
        self.state_bins = state_bins
        self.action_bins = action_bins
        self.max_battery = max_battery
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros(qtable_size)
        self.shape_weight = shape_weight
        self.buys = np.array([])

    def discretize_state(self, state):
        """	
        Discretizes the state into a tuple of indices.
        """
        # Extract variables from state
        battery_level = state[0] * 25
        hour = state[2]
        price = state[1]
        available = state[7]

        # Discretize the state variables
        battery_idx = np.digitize(battery_level, self.state_bins[0]) - 1
        time_idx = np.digitize(hour, self.state_bins[1]) - 1
        availability_idx = int(available)
        price_idx = np.digitize(price, self.state_bins[2]) - 1
        return battery_idx, time_idx, price_idx, availability_idx

    def discretize_action(self, action):
        """
        Discretizes the action into an index.
        """
        return np.digitize(action, self.action_bins) - 1

    def choose_action(self, state):
        """
        Chooses an action using epsilon-greedy.
        """
        # Ensure state is within valid range
        if not self.is_valid_state(state):
            return 0 
        
        # Choose random action with probability epsilon
        if np.random.random() < self.epsilon:
            return random.choice(self.action_bins)
        # Otherwise choose greedy action
        else:
            discretized_state = self.discretize_state(state)
            return self.action_bins[np.argmax(self.q_table[discretized_state])]

    def update(self, state, action, reward, next_state, last_price):
        """
        Updates the Q-table using Q-learning.
        """

        # Skip update if state is invalid
        if not self.is_valid_state(state) or not self.is_valid_state(next_state):
            return  
        
        # Discretize current state, next state, and action
        discretized_state = self.discretize_state(state)
        discretized_next_state = self.discretize_state(next_state)
        discretized_action = self.discretize_action(action)
        
        # Select best action for the next state
        best_next_action = np.argmax(self.q_table[discretized_next_state])

        # Reward shaping with increasing value
        shaped_reward = self.reward_shaping(state, next_state, action, last_price) * self.shape_weight
        
        # Calculate the TD target using the reward and Q-values of the next state, add reward shaping
        td_target = reward + shaped_reward + self.gamma * self.q_table[discretized_next_state + (best_next_action,)]
        
        # Calculate TD error
        td_error = td_target - self.q_table[discretized_state + (discretized_action,)]
        
        # Update Q-value using Q-learning update rule
        self.q_table[discretized_state + (discretized_action,)] += self.lr * td_error
        
    
    def update_epsilon(self):
        """Epsilon decay.

        Args:
            epsilon (_type_): The new epsilon value.
        """
        # Update epsilon value (exploration-exploitation tradeoff)
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)


    def is_valid_state(self, state):
        """"
        Checks if the state is valid.
        """
        # Get variables from state
        battery_level = state[0]
        hour = state[2]
        available = state[7]      
        # Implement logic to check if state is valid  
        return 0 <= battery_level <= self.max_battery and 1 <= hour <= 24 and 0 <= available <= 1
    
    def reward_shaping(self, state, next_state, action, last_price):
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

        buy_price = 0 if len(self.buys) == 0 else np.mean(self.buys)

        # If action is buying)
        if action > 0:
            # Compute the maximum amount of energy that can be bought
            max_buy = min(action, min(1,  (2 - battery_level) * 0.9))
            # If the agent buys between 3 am and 6 am 
            if 3 <= current_time <= 6:
                shaped_reward += 3
            # If the agent buys at a price less than 30
            if current_price <= 30:
                shaped_reward += 6
            # If the agent buys at a price greater than 70
            if current_price >= 70:
                shaped_reward -= 5
            if battery_level == 2:
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
            # Save the buy price
            self.buys = np.append(self.buys, current_price)
        # If action is selling
        elif action < 0:
            # Compute the maximum amount of energy that can be sold
            max_sell = max(action, -min(1, battery_level * 0.9))
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
            # Save the sell price
            self.buys = np.array([])
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
        return shaped_reward
        
class EMA:
  """Implements an exponential moving average cross strategy
  """
  def __init__(self, len_short: int, len_long: int, max_battery: int) -> None:
      """ Initialises the EMA strategy.

      Args:
          len_short (int): The length of the short EMA (in hours).
          len_long (int): The length of the long EMA (in hours).
          max_battery (int): The maximum battery capacity of the electric car.
      """
      # Constants
      self.len_short = len_short
      self.len_long = len_long
      self.max_battery = max_battery
      
      # Parameters
      self.sl_cross = 0
      self.ls_cross = 0
      self.short_ema = None
      self.long_ema = None
      self.short_ema_history = np.array([])
      self.long_ema_history = np.array([])
      self.action = None
  
  def calculate_ema(self, price: float, window: int, history: Type[np.ndarray]) -> Tuple[float, Type[np.ndarray]]:
      """ Calculates an EMA given a price value, the window span, the previous ema value, and a smoothing factor.

      Args:
          price (float): The price of the asset for the current time step.
          window (int): The window span of the EMA.
          history (Type[np.ndarray]): The history of the EMA.

      Returns:
          Tuple[float, Type[np.ndarray]]]: The EMA value and the updated history.
      """
      # If the history is empty, return the price
      if len(history) < 1:
          return price
      else:
          new = price * (2 / (window + 1)) + history[-1] * (1 - (2 / (window + 1)))
          return new

  def choose_action(self, state: list) -> float:
      """ Chooses an action for the current time step.

      Args:
          state (list): The current state of the environment.

      Returns:
          float: The action to take in terms of kW to buy or sell.
      """
      # Update the EMAs
      self.short_ema = self.calculate_ema(state[1], self.len_short, self.short_ema_history)
      self.long_ema = self.calculate_ema(state[1], self.len_long, self.long_ema_history)

      # Append the EMAs to the history
      self.short_ema_history = np.append(self.short_ema_history, self.short_ema)
      self.long_ema_history = np.append(self.long_ema_history, self.long_ema)
      # Choose the action
      # If the long EMA has not been calculated yet buy the max amount possible
      if len(self.long_ema_history) < self.len_long and state[2] == 4:
          self.action = (self.max_battery - state[0])
      # If the short EMA is below the long EMA, buy
      elif self.short_ema < self.long_ema and len(self.long_ema_history) >= self.len_long:
          self.ls_cross = 0
          self.sl_cross += 1
          if self.sl_cross >= 1 and 3 <= state[2] <= 6:
            self.action = (self.max_battery - state[0]) / 8.2
      # If the short EMA is above the long EMA, sell
      elif self.short_ema > self.long_ema and len(self.long_ema_history) >= self.len_long:
          self.sl_cross = 0
          self.ls_cross += 1
          if self.ls_cross >= 1:
              self.action = -state[0] / 2
      # Otherwise, do nothing
      else:
          self.action = 0
          
      return self.action

class BuyLowSellHigh:
  """Implements a buy low sell high strategy
  """
  def __init__(self, max_battery: int) -> None:
      """ Initialises the buy low sell high strategy.

      Args:
          max_battery (int): The maximum battery capacity of the electric car.
      """
      # Constants
      self.max_battery = max_battery

      # Parameters
      self.new_day = False
      self.action = None
      self.buy = None
      self.counter = 0
      self.s_counter = 0
  
  def choose_action(self, state: list) -> float:
      """ Chooses an action for the current time step.

      Args:
          price (float): The price of the asset for the current time step.

      Returns:
          float: The action to take in terms of kW to buy or sell.
      """
      # Reset the day boolean if it is a new day
      self.action = 0   

      # Buy in the morning between 3 and 6
      if 3 <= state[2] <= 6:
          # If it is a new day, buy one seventh of the max battery
          self.action += (self.max_battery - state[0]) / 8.2
          # Append the action to the history (price)
          self.buy = state[1]
      # Sell in the evening if the price is greater than  twice the buy price
      elif self.buy and state[1] >= 2 * self.buy: 
          # Check if car available
          if state[7]:
              # Add hour to counter
              self.counter += 1
              # If the counter is greater than 4, sell
              if self.counter > 4:
                  # Set action to sell
                  self.action = -state[0]
                  # If battery level is less than 25/0.9, reset the counter
                  if state[0] - (25 / 0.9) < 1:
                      self.counter = 0
          # If car not available, sell
          else:
              self.action = -state[0]
          
      return self.action

class DQNAgentLSTM:
    """
    This class represents the DQN agent.
    """
    def __init__(self, state_size, action_size, learning_rate=0.0001, gamma=0, batch_size=24):
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.device = device
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate
        self.model = LSTM_DQN(state_size, action_size, hidden_size=64, lstm_layers=1).to(device)
        self.target_model = LSTM_DQN(state_size, action_size, hidden_size=64, lstm_layers=1).to(device)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.epsilon = 0  # exploration rate
        self.epsilon_min = 0
        self.epsilon_decay = 0.95
        self.batch_size = batch_size

        self.train_step_counter = 0  # Counter to track training steps
        self.update_target_counter = 0  # Counter to track steps for updating target network

        self.buys = torch.from_numpy(np.array([], dtype=np.float32)).to(device)


    def remember(self, state, action, action_index, reward, next_state, done):
        """
        Function to store a transition in memory.
        """
        self.memory.append((state, action, action_index, reward, next_state, done))

    def choose_action(self, state_sequence, hidden_state=None):
        """Returns actions for given sequence of states as per current policy."""
        mid = 4
        #  Discretize action bins
        action_values = np.concatenate((
            np.linspace(-1, 0, mid, endpoint=False), np.linspace(0, .5, 6), np.array([.75, 1])
        ))  # Discretize actions (buy/sell amounts)

        if np.random.rand() > self.epsilon:  # Epsilon-greedy approach for exploitation
            with torch.no_grad():
                # Model forward pass with sequence of states
                q_values, hidden_state = self.model(state_sequence, hidden_state)

                # Get the index of the action with the highest Q-value
                action_index = np.argmax(q_values.cpu().data.numpy())

                # Return the action with the highest Q-value
                return action_index, action_values[action_index], hidden_state
        
        else:  # Exploration
            action_index = random.randrange(self.action_size)
            return action_index, action_values[action_index], hidden_state
        
    def replay(self, hidden_state=None):
        """
        Function to train the neural network on a batch of samples from memory.
        """
        self.train_step_counter += 1

        if len(self.memory) < self.batch_size or self.train_step_counter % 4 != 0:
            return  # Train only every 4th step and if enough samples

        minibatch = random.sample(self.memory, self.batch_size)
        
        # Convert sequences in minibatch to tensors and stack them
        state_sequences = torch.stack([torch.stack([torch.tensor(s, dtype=torch.float32) for s in m[0]]).unsqueeze(0) for m in minibatch])
        action_indices = torch.tensor([m[2] for m in minibatch], dtype=torch.int64).reshape(-1, 1).to(self.device)
        rewards = torch.tensor([m[3] for m in minibatch], dtype=torch.float32).to(self.device)
        next_state_sequences = torch.stack([torch.stack([torch.tensor(s, dtype=torch.float32) for s in m[4]]).unsqueeze(0) for m in minibatch])
        dones = torch.tensor([m[5] for m in minibatch], dtype=torch.float32).to(self.device)

        # Compute Q-values for current and next state sequences
        q_values, _ = self.model(state_sequences)
        next_q_values, _ = self.model(next_state_sequences)
        
        # Choose the best Q-value for next states
        max_next_q_values = next_q_values.max(dim=1)[0].detach()

        # Calculate target Q values
        Q_target = rewards + (self.gamma * max_next_q_values * (1 - dones))

        # Compute loss
        criterion = nn.HuberLoss()
        Q_expected = q_values.gather(1, action_indices)
        loss = criterion(Q_expected, Q_target.unsqueeze(1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network, if needed
        self.update_target_counter += 1
        if self.update_target_counter % 100000 == 0:
            self.target_model.load_state_dict(self.model.state_dict())

        # Update epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def reward_shaping(self, state, next_state, action, last_price):
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

        buy_price = 0 if len(self.buys) == 0 else self.buys.mean()

        # If action is buying)
        if action > 0:
            # Compute the maximum amount of energy that can be bought
            max_buy = min(action, min(1,  (2 - battery_level) * 0.9))
            # If the agent buys between 3 am and 6 am 
            if 3 <= current_time <= 6:
                shaped_reward += 3
            # If the agent buys at a price less than 30
            if current_price <= 30:
                shaped_reward += 6
            # If the agent buys at a price greater than 70
            if current_price >= 70:
                shaped_reward -= 5
            if battery_level == 2:
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
            self.buys = torch.cat((self.buys, torch.tensor([current_price]).to(self.device)))
        # If action is selling
        elif action < 0:
            # Compute the maximum amount of energy that can be sold
            max_sell = max(action, -min(1, battery_level * 0.9))
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
            self.buys.new_empty(0)
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
        return shaped_reward

    def load(self, name):
        """
        Function to load the model's weights.
        """
        self.model.load_state_dict(torch.load(name))

    def save(self, best_model, name):
        """
        Function to save the model's weights.
        """
        torch.save(best_model.state_dict(), name)

class LSTM_DQN(nn.Module):
    """
    This function represents the LSTM neural network used in the DQN algorithm.
    """
    def __init__(self, state_size, action_size, hidden_size=64, lstm_layers=1):
        super(LSTM_DQN, self).__init__()
        self.action_size = action_size
        self.lstm = nn.LSTM(state_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

    def forward(self, state, hidden_state=None):
        """
        This function computes the forward pass of the neural network.
        """
        # If the state is 2D (no batch dimension), add a batch dimension
        if state.dim() == 2:
            state = state.unsqueeze(0)  # Add batch dimension

        batch_size = state.size(0)
        
        # Initialize hidden state if not provided
        if hidden_state is None:
            hidden_state = self.init_hidden(batch_size)
        else:
            # Adjust the hidden state batch size if necessary
            hidden_state = (hidden_state[0][:, :batch_size, :], hidden_state[1][:, :batch_size, :])

        lstm_out, hidden_state = self.lstm(state, hidden_state)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        action_values = self.fc(lstm_out)
        
        return action_values, hidden_state


    def init_hidden(self, batch_size):
        """
        This function initializes the hidden state.
        """
        weight = next(self.parameters()).data
        hidden = (weight.new_zeros(self.lstm_layers, batch_size, self.hidden_size),
                  weight.new_zeros(self.lstm_layers, batch_size, self.hidden_size))
        return hidden
    
    

class LSTM_PolicyNetwork(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, lstm_layers=1):
        super(LSTM_PolicyNetwork, self).__init__()
        self.lstm = nn.LSTM(state_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc_mean = nn.Linear(hidden_size, action_size)
        self.fc_log_std = nn.Linear(hidden_size, action_size)   

    def forward(self, state, hidden_state=None):
        # state shape: (batch, sequence, features)
        if hidden_state is None:
            lstm_out, hidden_state = self.lstm(state)
        else:
            lstm_out, hidden_state = self.lstm(state, hidden_state)
        lstm_out = lstm_out[:, -1, :]  # Take the output of the last time step
        action_mean = self.fc_mean(lstm_out)  # Linear layer for mean
        action_log_std = self.fc_log_std(lstm_out)  # Linear layer for log_std
        return action_mean, action_log_std.exp(), hidden_state


    def init_hidden(self, device, batch_size):
        # Initializes the hidden state
        # This depends on the number of LSTM layers and whether it's bidirectional
        weight = next(self.parameters()).data
        hidden = (weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_().to(device),
                  weight.new(self.lstm.num_layers, batch_size, self.lstm.hidden_size).zero_().to(device))
        return hidden