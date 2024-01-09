import gym
from typing import Type
import numpy as np
import pandas as pd
from gym import spaces

class ElectricCarEnv(gym.Env):
    """ Implements the gym interface for the electric car trading problem.
    """
    def __init__(self):
        """ Initializes the environment.
        """
        super(ElectricCarEnv, self).__init__()

        # Constants

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
        self.price_data = pd.read_excel('train.xls')
        self.current_step = 0

    def step(self, action):
        # Update the battery level based on the action and efficiency
        energy_change = action * self.efficiency
        self.battery_level = min(max(self.battery_level + energy_change, 0), self.max_battery)

        # Calculate reward (profit from buying/selling electricity)
        price = self.get_current_price()
        reward = -action * price if action < 0 else -2 * action * price

        # Update state
        self.time_of_day += 1
        
        # From 6 am to 8 pm, the car is unavailable
        if 6 <= self.time_of_day <= 20:
            self.car_available = 0

        if self.time_of_day >= 24:
            self.time_of_day = 0
            self.current_step += 1
            # Handle car availability and battery level
            self.car_available = np.random.choice([0, 1], p=[0.5, 0.5])
            if not self.car_available:
                self.battery_level = max(self.battery_level - 20, self.min_required_battery)

        # Check if the battery level is below the minimum required at 8 am
        if self.time_of_day == 8 and self.battery_level < self.min_required_battery:
            self.battery_level = self.min_required_battery

        # Update the state
        self.state = [self.battery_level, self.time_of_day, self.car_available]

        # Check if the episode is done
        done = self.current_step >= len(self.price_data)

        return np.array(self.state), reward, done, {}

    def reset(self):
        # Start with a minimum battery level
        self.battery_level = self.min_required_battery
        self.time_of_day = 0
        self.car_available = 1
        self.current_step = 0
        self.state = [self.battery_level, self.time_of_day, self.car_available]
        return np.array(self.state)

    def get_current_price(self):
        # Return the current electricity price
        return self.price_data.iloc[self.current_step]['price']

env = ElectricCarEnv()
