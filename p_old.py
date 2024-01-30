from data.data_vis import clean_data
import pandas as pd
import numpy as np

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
    for i in range(min, max):
        # Compute the exponential moving averages with the specified span
        data[f'{i}EMA'] = pd.Series(df).ewm(span=i, adjust=False).mean()

    for n in [50, 100, 200]:
        # Compute the moving averages with the specified window
        data[f'{n}MA'] = pd.Series(df).rolling(window=n, min_periods=1).mean()
        data[f'{n * 4}MA'] = pd.Series(df).rolling(window=n*4, min_periods=1).mean()
        data[f'{n * 24}MA'] = pd.Series(df).rolling(window=n*24, min_periods=1).mean()
    
    # Compute the expanding average
    data['ExMA'] = pd.Series(df).expanding(min_periods=1).mean()
    # Compute the expanding median
    data['ExMedian'] = pd.Series(df).expanding(min_periods=1).median()
    # Compute the expanding min
    data['ExMin'] = pd.Series(df).expanding(min_periods=1).min()
    # Compute the expanding max
    data['ExMax'] = pd.Series(df).expanding(min_periods=1).max()
    # Compute the expanding std
    data['ExStd'] = pd.Series(df).expanding(min_periods=1).std().fillna(0)
    # Compute the expanding var
    data['ExVar'] = pd.Series(df).expanding(min_periods=1).var().fillna(0)

    return data

def season(month: int) -> int:
    """ Returns the season of the month.

    Args:
        month (int): The month of the year.

    Returns:
        int: The season of the month.
    """
    if month in [1, 2, 12]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    elif month in [9, 10, 11]:
        return 4

def weekend(day: int) -> int:
    """ Returns 1 if the day is a weekend, 0 otherwise.

    Args:
        day (int): The day of the week.

    Returns:
        int: 1 if the day is a weekend, 0 otherwise.
    """
    if day in [5, 6]:
        return 1
    else:
        return 0
    
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

    new_data = {
        'Season': np.repeat(date_df['Month'].apply(season), 24).reset_index(drop=True),
        'Weekend': np.repeat(date_df['Day'].apply(weekend), 24).reset_index(drop=True),
    }

    # Compute the moving averages
    new_data = compute_stats(new_data, df, 3, 13)

    # Assign the new_data to the new_df
    new_df = pd.DataFrame(new_data)

    # Save the new_df to a new excel file
    new_df.to_excel(save_to, index=False)

# Example usage
create_features('data/train.xlsx', 'data/f_train.xlsx')