import numpy as np
import random
from ElectricCarEnv import ElectricCarEnv
from typing import Type, Tuple

class QLearningAgent:
    """
    Implements a simple tabular Q-learning agent for the electric car trading problem.
    """	
    def __init__(self, state_bins, action_bins, learning_rate=0.01, discount_factor=0.99, epsilon=1, epsilon_decay=0.995, min_epsilon=0, max_battery=50):
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
            return 0 #TODO decide what to do for invalid action
         
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
  def __init__(self, len_short: int, len_long: int, alpha_short: float, alpha_long: float, max_battery: int) -> None:
      """ Initialises the EMA strategy.

      Args:
          len_short (int): The length of the short EMA (in hours).
          len_long (int): The length of the long EMA (in hours).
          alpha_short (float): The alpha value for the short EMA.
          alpha_long (float): The alpha value for the long EMA.
      """
      # Constants
      self.len_short = len_short
      self.len_long = len_long
      self.alpha_short = alpha_short
      self.alpha_long = alpha_long
      self.max_battery = max_battery

      self.short_ema = None
      self.long_ema = None
      self.short_ema_history = np.array([])
      self.long_ema_history = np.array([])
      self.action = None
  
  def calculate_ema(self, price: float, window: int, history: Type[np.ndarray], alpha: float) -> Tuple[float, Type[np.ndarray]]:
      """ Calculates an EMA given a price value, the window span, the previous ema value, and a smoothing factor.

      Args:
          price (float): The price of the asset for the current time step.
          window (int): The window span of the EMA.
          history (Type[np.ndarray]): The history of the EMA.
          alpha (float): The smoothing factor.

      Returns:
          Tuple[float, Type[np.ndarray]]]: The EMA value and the updated history.
      """
      if len(history) < 1:
          return price, np.append(history, price)
      else:
          new = price * (alpha / (window + 1)) + history[-1] * (1 - (alpha / (window + 1)))
          return new, np.append(history, new)
  
  def choose_action(self, price: float, state: list) -> float:
      """ Chooses an action for the current time step.

      Args:
          price (float): The price of the asset for the current time step.

      Returns:
          float: The action to take in terms of kW to buy or sell.
      """
      # Update the EMAs
      self.short_ema, self.short_ema_history = self.calculate_ema(price, self.len_short, self.short_ema_history, self.alpha_short)
      self.long_ema, self.long_ema_history = self.calculate_ema(price, self.len_long, self.long_ema_history, self.alpha_long)

      # Append the EMAs to the history
      self.short_ema_history = np.append(self.short_ema_history, self.short_ema)
      self.long_ema_history = np.append(self.long_ema_history, self.long_ema)
      
      # Choose the action
      if not len(self.long_ema_history) < self.len_long:
          if state[1] < 8 or state[1] > 18:
              self.action = self.max_battery - state[0]

          if self.short_ema < self.long_ema:
              self.action *= -1
          elif self.short_ema == self.long_ema:
              self.action = 0
      else:
          self.action = 0
          
      return self.action
  
  def update(self, price):
      pass
