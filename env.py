import gym
from typing import Tuple
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
        self.data = pd.read_csv('train.csv')

        # Initialize the state
        self.current_step = 0
        self.time_of_day = 1
        
    
    def two_day_availability(self) -> None:
        """ Generates a random two-day availability schedule for the car.
        """
        # Generate a random two-day availability schedule
        availability = np.random.choice([0, 1], size=2, p=[0.5, 0.5])
        if availability[0] == 1:
            self.availabilities = [1] * 31 + [0] * 11 + [1] * 6
        else:
            self.availabilities = [1] * 7 + [0] * 11 + [1] * 30

    def step(self, action: float) -> Tuple[np.array, float, bool, dict]:
        """ Implements the step function for the environment.

        Args:
            action (float): The amount of electricity to buy or sell (negative for selling)

        Returns:
            Tuple[np.array, float, bool, dict]: The next state, the reward, whether the episode is done, and additional information
        """
        # Update the battery level based on the action and efficiency
        energy_change = action * self.efficiency
        self.battery_level = min(max(self.battery_level + energy_change, 0), self.max_battery)

        # Calculate reward (profit from buying/selling electricity)
        price = self.get_current_price()
        reward = action * price if action > 0 else action * price / self.efficiency

        # Update state
        self.time_of_day += 1
        self.current_step += 1
        if self.time_of_day > 24:
            self.time_of_day = 1
            # Handle car availability and battery level
            self.car_available = self.availabilities.pop(0)
            if len(self.availabilities) == 0: 
                self.availabilities = self.two_day_availability()
            if not self.car_available:
                self.battery_level = max(self.battery_level - 20, self.min_required_battery)

        # Check if the battery level is below the minimum required at 7 am
        if self.time_of_day == 7 and self.battery_level < self.min_required_battery:
            # Decrease the reward by the cost of charging the battery to the minimum required
            self.reward -= (self.min_required_battery - self.battery_level) * price / 0.9
            self.battery_level = self.min_required_battery

        # Update the state
        self.state = [self.battery_level, self.time_of_day, self.car_available]

        # Check if the episode is done
        done = self.current_step >= len(self.price_data)

        return np.array(self.state), reward, done, {}

    def reset(self) -> None:
        """ Resets the environment.
        """
        # Start with a minimum battery level
        self.battery_level = self.min_required_battery
        # Start at the beginning of the price data
        self.time_of_day = 1
        self.current_step = 0
        # Generate a random two-day availability schedule
        self.car_available = self.two_day_availability()
        # Initialize the state
        self.state = [self.battery_level, self.time_of_day, self.car_available]
        return np.array(self.state)

    def get_current_price(self) -> float:
        """ Returns the current electricity price.
        """
        #TODO IMPLEMENT
        # Return the current electricity price
        return self.data.iloc[self.current_step]['price']

env = ElectricCarEnv()
