import numpy as np
import random
from typing import Type, Tuple
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque

class QLearningAgent:
    """
    Implements a simple tabular Q-learning agent for the electric car trading problem.
    """	
    def __init__(self, state_bins, action_bins, learning_rate=0.01, discount_factor=0.95, epsilon=1, epsilon_decay=0.995, min_epsilon=0, max_battery=50):
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
        battery, time, availability = state
        battery_idx = np.digitize(battery, self.state_bins[0]) - 1
        time_idx = np.digitize(time, self.state_bins[1]) - 1
        availability_idx = int(availability)
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
        battery, time, availability = state
        return 0 <= battery <= self.max_battery and 1 <= time <= 24 and 0 <= availability <= 1
    
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
  
  def percent_difference(self) -> float:
      """ Calculates the percent difference between the short and long EMAs.

      Returns:
          float: The percent difference between the short and long EMAs.
      """
      return abs((self.short_ema - self.long_ema)) / self.long_ema

  def choose_action(self, price: float, state: list) -> float:
      """ Chooses an action for the current time step.

      Args:
          price (float): The price of the asset for the current time step.

      Returns:
          float: The action to take in terms of kW to buy or sell.
      """
      # Update the EMAs
      self.short_ema = self.calculate_ema(price, self.len_short, self.short_ema_history)
      self.long_ema = self.calculate_ema(price, self.len_long, self.long_ema_history)

      # Append the EMAs to the history
      self.short_ema_history = np.append(self.short_ema_history, self.short_ema)
      self.long_ema_history = np.append(self.long_ema_history, self.long_ema)
      
      # Choose the action
      # If the long EMA has not been calculated yet buy the max amount possible
      if len(self.long_ema_history) < self.len_long and state[1] == 4:
          self.action = -(self.max_battery - state[0])
      # If the short EMA is below the long EMA, buy
      elif self.short_ema < self.long_ema:
          self.ls_cross = 0
          self.sl_cross += 1
          if self.sl_cross > 1 and 3 <= state[1] <= 6:
            self.action = -(self.max_battery - state[0])
      # If the short EMA is above the long EMA, sell
      elif self.short_ema > self.long_ema:
          self.sl_cross = 0
          self.ls_cross += 1
          if self.ls_cross > 6:
              self.action = state[0]
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
      self.history = np.array([])
      self.count = 0
  
  def choose_action(self, price: float,  state: list) -> float:
      """ Chooses an action for the current time step.

      Args:
          price (float): The price of the asset for the current time step.

      Returns:
          float: The action to take in terms of kW to buy or sell.
      """
      # Reset the day boolean if it is a new day
      if state[1] == 1:
          self.new_day = True
      self.action = 0   

      # Choose the action 

      # Buy in the morning
      if 4 <= state[1] <= 5:
          # If it is a new day, buy one seventh of the max battery
          self.action -= (self.max_battery - state[0]) / 2
          self.amount = self.action
          # Append the action to the history
          self.buy = 2 * price
      # Sell in the evening
      elif 7 <= state[1] <= 20 and price >= self.buy:
          self.action = state[0]
          
      return self.action
  
class Sell:
    def __init__(self):
        self.max_battery = 50

    def choose_action(self, state: list) -> float:
        return state[0]

class QNetwork(nn.Module):
    """
    This class represents the neural network used to approximate the Q-function.
    """
    def __init__(self, state_size, action_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

class DQNAgent:
    """
    This class represents the DQN agent.
    """
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=10000)
        self.gamma = 0.95  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.batch_size = 64
        self.learning_rate = 0.001

        self.model = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)

    def remember(self, state, action, reward, next_state, done):
        """
        Function to store a transition in memory.
        """
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        """Returns actions for given state as per current policy."""
        state_vector = [state[0] / 50, state[1] / 24, state[2]]  # Assuming state is a list or array
        state_tensor = torch.FloatTensor(state_vector).unsqueeze(0)

        if np.random.rand() > self.epsilon:  # Epsilon-greedy approach
            with torch.no_grad():
                action_values = self.model(state_tensor)
                return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.randrange(self.action_size)

    def replay(self):
        """
        Function to train the neural network on a batch of samples from memory.
        """
        if len(self.memory) < self.batch_size:
            return  # Do not train if not enough samples

        minibatch = random.sample(self.memory, self.batch_size)
        
        # Extract information from each memory and convert to numpy arrays
        # Ensure the array is of type float for subsequent operations
        states = np.array([m[0] for m in minibatch], dtype=np.float32)
        actions = np.array([m[1] for m in minibatch], dtype=np.int64).reshape(-1, 1)
        rewards = np.array([m[2] for m in minibatch], dtype=np.float32)
        next_states = np.array([m[3] for m in minibatch], dtype=np.float32)
        dones = np.array([m[4] for m in minibatch], dtype=np.float32)

        # Normalize states and next states
        states[:, 0] /= 50.0  # Make sure to use float division
        states[:, 1] /= 24.0  # Make sure to use float division
        next_states[:, 0] /= 50
        next_states[:, 1] /= 24

        # Convert numpy arrays to tensors
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Compute Q values for current states
        Q_expected = self.model(states).gather(1, actions)

        # Compute Q values for next states and calculate target
        with torch.no_grad():
            Q_next = self.model(next_states).max(1)[0].detach()
            Q_target = rewards + (self.gamma * Q_next * (1 - dones))

        # Calculate loss
        loss = nn.MSELoss()(Q_expected, Q_target.unsqueeze(1))

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

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