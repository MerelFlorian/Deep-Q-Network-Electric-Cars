import numpy as np

def clip(action: float, state: np.ndarray) -> float:
    """ Clip the action value to the minimum/maximum power

    Args:
        action (float): The action value.
        state (np.ndarray): The observations available to the agent.

    Returns:
        float: The clipped action value.
    """
    # Clip the action value to the minimum/maximum power
    max_action = min(action, min(25,  (50 - state[0]) * 0.9)) if action > 0 else 0
    min_action = max(action, -min(25, state[0] * 0.9)) if action < 0 else 0
    return np.clip(action, min_action, max_action) / 25