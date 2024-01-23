from data.data_vis import clean_data
import pandas as pd

PATH = "data/train.xlsx"
SAVE_TO = "data/features.xlsx"

def compute_ma(data: dict, df: pd.DataFrame, min: int, max: int) -> dict:
    """ Computes the moving average of the data and adds it to a dictionary of columns.

    Args:
        data (dict): The dictionary containing the columns to compute the moving average
        df (pd.DataFrame): The dataframe to compute the moving averages from.
        min (int): The minimum number of days to compute the moving average from.
        max (int): The maximum number of days to compute the moving average from.

    Returns:
        dict: The dictionary containing the moving averages.
    """
    # Flatten the DataFrame into a single Series
    flat_series = df.drop('date', axis=1).values.flatten()

    for i in range(min, max):
        # Compute the rolling average with the specified window
        ma = pd.Series(flat_series).rolling(window=i, min_periods=1).mean()
        ema = pd.Series(flat_series).ewm(span=i, adjust=False).mean()
        
        # Store the results in the dictionary
        data[f'{i}MA'] = ma.values
        data[f'{i}EMA'] = ema.values

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

    # Define columns for featureset
    features = ["3MA", "4MA", "5MA", "6MA", "7MA", "8MA", "10MA", "12MA",
                "3EMA", "4EMA", "5EMA", "6EMA", "7EMA", "8EMA", "10EMA", "12EMA",
                "Season", "Weekend", 'Mean', 'Median', 'Min', 'Max', 'Std', 'Var']
    
    # Create an empty DataFrame with these columns
    new_df = pd.DataFrame(columns=features)

    # Concatenate all hourly prices into a single series for rolling calculations
    hourly_prices = df[['H1', 'H2', 'H3', 'H4', 'H5', 'H6', 'H7', 'H8', 'H9', 'H10', 
                        'H11', 'H12', 'H13', 'H14', 'H15', 'H16', 'H17', 'H18', 
                        'H19', 'H20', 'H21', 'H22', 'H23', 'H24']].values.flatten()

    # Convert to a Pandas Series for rolling calculations
    hourly_prices_series = pd.Series(hourly_prices)

    new_data = {
        'Season': df['Month'].apply(season),
        'Weekend': df['Day'].apply(weekend),
        'Mean':  hourly_prices_series.rolling(window=len(hourly_prices_series), min_periods=1).mean()[:len(df)],
        'Median': hourly_prices_series.rolling(window=len(hourly_prices_series), min_periods=1).median()[:len(df)],
        'Min': hourly_prices_series.rolling(window=len(hourly_prices_series), min_periods=1).min()[:len(df)],
        'Max': hourly_prices_series.rolling(window=len(hourly_prices_series), min_periods=1).max()[:len(df)],
        'Std': hourly_prices_series.rolling(window=len(hourly_prices_series), min_periods=1).std()[:len(df)],
        'Var': hourly_prices_series.rolling(window=len(hourly_prices_series), min_periods=1).var()[:len(df)],
    }

    # Add the features to the DataFrame
    for feature in features:
        new_df[feature] = new_data[feature]

    # Drop the NaN values
    new_df = new_df.dropna()

    # Compute the moving averages
    new_data = compute_ma(new_data, df, 3, 13)

    # Create a new DataFrame
    new_df = pd.concat(new_data.values(), axis=1)

    # Save the new_df to a new excel file
    new_df.to_excel(save_to, index=False)

if __name__ == "__main__":

    main(PATH, SAVE_TO)