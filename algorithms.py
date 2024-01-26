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
    def __init__(self, state_bins, action_bins, learning_rate=0.01, discount_factor=0.5, epsilon=1, epsilon_decay=0.995, min_epsilon=0, max_battery=50):
        self.state_bins = state_bins
        self.action_bins = action_bins
        self.max_battery = max_battery
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.q_table = np.zeros(shape=(len(state_bins[0]), len(state_bins[1]), len(state_bins[2]), len(action_bins)))

    def discretize_state(self, state):
        """	
        Discretizes the state into a tuple of indices.
        """
        battery_level, _, hour, available , _, _, _, _ = state
        battery_idx = np.digitize(battery_level, self.state_bins[0]) - 1
        time_idx = np.digitize(hour, self.state_bins[1]) - 1
        availability_idx = int(available)
        return battery_idx, time_idx, availability_idx

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
         
        if np.random.random() < self.epsilon:
            return random.choice(range(len(self.action_bins)))
        else:
            discretized_state = self.discretize_state(state)
            return np.argmax(self.q_table[discretized_state])

    def update(self, state, action, reward, next_state):
        """
        Updates the Q-table using Q-learning.
        """
        # Skip update if state is invalid
        if not self.is_valid_state(state) or not self.is_valid_state(next_state):
            return  
        
        discretized_state = self.discretize_state(state)
        discretized_next_state = self.discretize_state(next_state)
        discretized_action = self.discretize_action(action)
        best_next_action = np.argmax(self.q_table[discretized_next_state])
        td_target = reward + self.gamma * self.q_table[discretized_next_state + (best_next_action,)]
        td_error = td_target - self.q_table[discretized_state + (discretized_action,)]
        self.q_table[discretized_state + (discretized_action,)] += self.lr * td_error
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.min_epsilon)

    def is_valid_state(self, state):
        """"
        Checks if the state is valid.
        """
        # Implement logic to check if state is valid
        battery_level, price, hour, available , day_of_week, day_of_year, month, year = state
        return 0 <= battery_level <= self.max_battery and 1 <= hour <= 24 and 0 <= available <= 1
    
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
  
class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, activation_fn=torch.relu):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, action_size)
        self.activation_fn = activation_fn

        # Apply He initialization
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity=activation_fn.__name__)
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity=activation_fn.__name__)
        init.kaiming_normal_(self.fc3.weight, mode='fan_in', nonlinearity=activation_fn.__name__)
        init.kaiming_normal_(self.fc4.weight, mode='fan_in', nonlinearity=activation_fn.__name__)
        init.kaiming_normal_(self.fc5.weight, mode='fan_in', nonlinearity='linear')

    def forward(self, state):
        x = self.activation_fn(self.fc1(state))
        x = self.activation_fn(self.fc2(x))
        x = self.activation_fn(self.fc3(x))
        x = self.activation_fn(self.fc4(x))
        return self.fc5(x)


class DQNAgent:
    """
    This class represents the DQN agent.
    """
    def __init__(self, state_size, action_size, learning_rate=0.0001, gamma=0.95, batch_size=24):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=1000)
        self.gamma = gamma  # discount factor
        self.learning_rate = learning_rate
        # self.model = QNetwork(state_size, action_size, activation_fn)
        # self.target_model = QNetwork(state_size, action_size, activation_fn)
        self.model = LSTM_DQN(state_size, action_size, hidden_size=256, lstm_layers=1)
        self.target_model = LSTM_DQN(state_size, action_size, hidden_size=256, lstm_layers=1)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = batch_size

        self.train_step_counter = 0  # Counter to track training steps
        self.update_target_counter = 0  # Counter to track steps for updating target network


    def remember(self, state, action, reward, next_state, done):
        """
        Function to store a transition in memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state_sequence, hidden_state=None):
        """Returns actions for given sequence of states as per current policy."""
        
        if np.random.rand() > self.epsilon:  # Epsilon-greedy approach for exploitation
            with torch.no_grad():
                # Model forward pass with sequence of states
                q_values, hidden_state = self.model(state_sequence, hidden_state)
                return np.argmax(q_values.cpu().data.numpy()), hidden_state
        else:  # Exploration
            return random.randrange(self.action_size), hidden_state
        
    def replay(self):
        """
        Function to train the neural network on a batch of samples from memory.
        """
        self.train_step_counter += 1

        if len(self.memory) < self.batch_size or self.train_step_counter % 4 != 0:
            return  # Train only every 4th step and if enough samples

        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract information from each memory and convert to numpy arrays
        states = np.array([m[0] for m in minibatch], dtype=np.float32)
        actions = np.array([m[1] for m in minibatch], dtype=np.int64).reshape(-1, 1)
        rewards = np.array([m[2] for m in minibatch], dtype=np.float32)
        next_states = np.array([m[3] for m in minibatch], dtype=np.float32)
        dones = np.array([m[4] for m in minibatch], dtype=np.float32)

        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Get Q values for current states
        Q_values, _ = self.model(states)
        Q_values = Q_values.repeat(self.batch_size, 1)
        Q_expected = Q_values.gather(1, actions)

        # Get Q values for next states
        Q_values_next, _ = self.model(next_states)
        Q_next = Q_values_next.max(1)[0].detach()
    
        # Calculate target Q values
        Q_target = rewards + (self.gamma * Q_next * (1 - dones))

        # Compute loss
        criterion = nn.HuberLoss()
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


    def load(self, name):
        """
        Function to load the model's weights.
        """
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        """
        Function to save the model's weights.
        """
        torch.save(self.model.state_dict(), name)

class LSTM_DQN(nn.Module):
    def __init__(self, state_size, action_size, hidden_size=64, lstm_layers=1):
        super(LSTM_DQN, self).__init__()
        self.action_size = action_size
        self.lstm = nn.LSTM(state_size, hidden_size, num_layers=lstm_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, action_size)
        self.hidden_size = hidden_size
        self.lstm_layers = lstm_layers

    def forward(self, state, hidden_state=None):
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