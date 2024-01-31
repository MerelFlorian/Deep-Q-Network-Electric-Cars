import numpy as np
import pandas as pd

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

def clean_data(file: str) -> pd.DataFrame:
    """Clean the data. Removes NAN values and removes all columns after 24th column. 
    Removes empty rows. Dates are converted to datetime format. Columns are renamed to H1, H2, H3.

    Args:
        file (str): Path to excel file

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """

    # Import data as pandas dataframe
    df = pd.read_excel(file)
    # Remove all columns after 24th column
    df = df.iloc[:, :25]
    # Remove all rows with NaN values
    df = df.dropna()
    # Rename columns Hourn to Hn
    for i in range(1, 25):
        if i < 10:
            df.rename(columns={f'Hour 0{i}': f'H{i}'}, inplace=True)
        else:
            df.rename(columns={f'Hour {i}': f'H{i}'}, inplace=True)
    # Rename PRICES to date
    df.rename(columns={'PRICES': 'date'}, inplace=True)
    # Convert date column to datetime
    df['date'] = pd.to_datetime(df['date'], format='mixed')
    return df

def compute_stats(data: dict, df: pd.DataFrame, min: int, max: int) -> dict:
    """ Computes the moving average of the data and adds it to a dictionary of columns.

    Args:
        data (dict): The dictionary containing the columns to compute the moving average
        df (pd.DataFrame): The dataframe to compute the moving averages from.
        min (int): The minimum number of days to compute the moving average from.
        max (int): The maximum number of days to compute the moving average from.

    Returns:
        dict: The dictionary containing the moving averages.
    """
    for i in [min, max]:
        # Compute the exponential moving averages with the specified span
        data[f'{i}EMA'] = pd.Series(df).ewm(span=i, adjust=False).mean()

    return data

def create_features(path: str, save_to="features.xlsx") -> None:
    """ Creates a new excel file with features extracted from the dataset.
        NOTE: This function does not create "polluting" features like global statistics.

    Args:
        path (str): The path to the dataset.
    """
    # Load the dataset
    df = clean_data(path)

    # Convert 'date' column to DateTime
    df['date'] = pd.to_datetime(df['date'])

    # Extract day of the week and month
    df['Day'] = df['date'].dt.weekday + 1  # Monday=1, Tuesday=2, etc.
    df['Month'] = df['date'].dt.month 

    date_df = df[['date', 'Day', 'Month']]

    # Concatenate all hourly prices into a single series for rolling calculations
    df = df.drop(columns=['date', 'Day', 'Month']).values.flatten()

    new_data = {}

    # Compute the moving averages
    new_data = compute_stats(new_data, df, 3, 7)

    # Assign the new_data to the new_df
    new_df = pd.DataFrame(new_data)

    # Save the new_df to a new excel file
    new_df.to_excel(save_to, index=False)