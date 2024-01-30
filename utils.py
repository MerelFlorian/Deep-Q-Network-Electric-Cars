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

def save_best_q(best_q_table: np.ndarray, highest_reward: float, episode: int, filename: str = 'best_q_table.npy'):
    """ Function to save the best Q-table.

    Args:
        best_q_table (np.ndarray): The best Q-table.
        highest_reward (float): The highest reward achieved.
        episode (int): The episode the best Q-table was achieved in.
        filename (str, optional): The filename to save the Q-table to. Defaults to 'best_q_table.npy'.

    Returns:
        None
    """
    np.save(filename, best_q_table)
    print(f"Best Q-table saved from episode {episode} with total reward: {highest_reward}")