import gym
from typing import Tuple
import numpy as np
import pandas as pd
from gym import spaces

class ElectricCarEnv(gym.Env):
    """ Implements the gym interface for the electric car trading problem.
    """
    def __init__(self) -> None:
        """ Initializes the environment.
        """
        super(ElectricCarEnv, self).__init__()

        # Constants:
        # Battery capacity in kWh
        self.max_battery = 50
        # Minimum required battery level in kWh
        self.min_required_battery = 20
        # Efficiency of the battery for charging/discharging
        self.efficiency = 0.9
        # Maximum charging/discharging power in kW
        self.max_power = 25

        # State space: [current battery level, time of day, car availability]
        self.observation_space = spaces.Box(low=np.array([0, 0, 0]), high=np.array([self.max_battery, 24, 1]), dtype=np.float32)

        # Action space: amount of electricity to buy or sell (negative for selling)
        self.action_space = spaces.Box(low=-self.max_power, high=self.max_power, dtype=np.float32)

        # Load electricity price data
        self.data = pd.read_csv('data/train_clean.csv')

        # Initialize the state
        self.current_step = 0
        self.time_of_day = 1

    def step(self, action: float) -> Tuple[np.ndarray, float, bool, dict]:
        """ Implements the step function for the environment.

        Args:
            action (float): The amount of electricity to buy or sell (negative for selling)

        Returns:
            Tuple[np.array, float, bool, dict]: The next state, the reward, whether the episode is done, and additional information
        """
        # Update the time
        self.time_of_day += 1
        # Check if the day is over
        if self.time_of_day > 24:
            # Update the current step
            self.current_step += 1
            # Reset the time of day to 1AM
            self.time_of_day = 1
            # Randomly decide if the car is available for the new day
            self.car_available = np.random.choice([0, 1])
        
        action = min(action, self.max_power) if action > 0 else max(action, -self.max_power)
        
        # From 8AM to 6PM, unavailable cars can't be charged
        if not (8 <= self.time_of_day <= 18 and not self.car_available):
            energy_change = (action / self.efficiency if action > 0 else action)
            # Update the battery level based on the action and efficiency
            self.battery_level = min(max(self.battery_level + energy_change, 0), self.max_battery)

            # Calculate reward (profit from buying/selling electricity)
            price = self.get_current_price()
            #Note: - for action means buying, + for action means selling
            reward = (action * price if action > 0 else 2 * action * price / 0.9) / 1000
        else:
            reward = 0

        # After the car returns from 8AM-6PM, the battery level decreases by 20 kWh
        if self.time_of_day == 19 and not self.car_available:
            self.battery_level = max(self.battery_level - 20, self.min_required_battery)

        # Check if the battery level is below the minimum required at 7 am
        if self.time_of_day == 7 and self.battery_level < self.min_required_battery:
            # Decrease the reward by the cost of charging the battery to the minimum required
            reward -= (self.min_required_battery - self.battery_level) * 2 * price / 1000 / 0.9
            self.battery_level = self.min_required_battery

        # Update the state
        self.state = np.array([self.battery_level, self.time_of_day, self.car_available])

        # Check if the episode is done
        done = self.current_step == len(self.data) - 1

        #self.revenue += reward

        return self.state, reward, done, {'step':self.current_step}

    def reset(self) -> np.ndarray:
        """ Resets the environment.
        """
        # Start with a minimum battery level
        self.battery_level = self.min_required_battery
        # Reset the time of day and current step
        self.time_of_day = 1
        self.current_step = 0
        # Generate a random car availability
        self.car_available = np.random.choice([0, 1])
        # Initialize the state
        self.state = np.array([self.battery_level, self.time_of_day, self.car_available])

        return self.state

    def get_current_price(self) -> float:
        """ Returns the current electricity price.
        """
        return self.data.iloc[self.current_step]["H" + str(self.time_of_day)]