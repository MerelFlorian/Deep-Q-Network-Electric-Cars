from data.data_vis import clean_data
import pandas as pd

PATH = "data/train.xlsx"
SAVE_TO = "data/features.xlsx"

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
        # Compute the rolling average with the specified window
        data[f'{i}MA'] = pd.Series(df).rolling(window=i, min_periods=1).mean()
        data[f'{i}EMA'] = pd.Series(df).ewm(span=i, adjust=False).mean()
    
    # Compute the expanding average
    data['ExMA'] = pd.Series(df).expanding(min_periods=1).mean()
    # Compute the expanding median
    data['ExMedian'] = pd.Series(df).expanding(min_periods=1).median()
    # Compute the expanding min
    data['ExMin'] = pd.Series(df).expanding(min_periods=1).min()
    # Compute the expanding max
    data['ExMax'] = pd.Series(df).expanding(min_periods=1).max()
    # Compute the expanding std
    data['ExStd'] = pd.Series(df).expanding(min_periods=1).std()
    # Compute the expanding var
    data['ExVar'] = pd.Series(df).expanding(min_periods=1).var()

    return data

def season(month: int) -> int:
    if month in [1, 2, 12]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8]:
        return 3
    elif month in [9, 10, 11]:
        return 4

def weekend(day: int) -> int:
    if day in [5, 6]:
        return 1
    else:
        return 0
    
def main(path: str, save_to:str) -> None:
    # Load the dataset
    df = clean_data(path)

    # Convert 'date' column to DateTime
    df['date'] = pd.to_datetime(df['date'])

    # Extract day of the week and month
    df['Day'] = df['date'].dt.weekday + 1  # Monday=1, Tuesday=2, etc.
    df['Month'] = df['date'].dt.month 

    date_df = df[['date', 'Day', 'Month']]

    # Define columns for featureset
    features = ["3MA", "4MA", "5MA", "6MA", "7MA", "8MA", "10MA", "12MA",
                "3EMA", "4EMA", "5EMA", "6EMA", "7EMA", "8EMA", "10EMA", "12EMA",
                "Season", "Weekend"]
    
    # 'Mean', 'Median', 'Min', 'Max', 'Std', 'Var'

    # Create an empty DataFrame with these columns
    new_df = pd.DataFrame(columns=features)

    # Concatenate all hourly prices into a single series for rolling calculations
    df = df.drop(columns=['date', 'Day', 'Month']).values.flatten()

    new_data = {
        'Season': date_df['Month'].apply(season),
        'Weekend': date_df['Day'].apply(weekend)
    }

    # Compute the moving averages
    new_data = compute_stats(new_data, df, 3, 13)

    # Assign the new_data to the new_df
    new_df = new_df.assign(**new_data)

    # Drop the NaN values
    new_df = new_df.dropna()

    # Save the new_df to a new excel file
    new_df.to_excel(save_to, index=False)

if __name__ == "__main__":

    main(PATH, SAVE_TO)