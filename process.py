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

def main(path: str, save_to: str) -> None:
    # Load the dataset
    df = clean_data(path)
    new_data = {}

    # Compute the moving averages
    new_data = compute_ma(new_data, df, 3, 13)

    # Create a new DataFrame
    new_df = pd.concat(new_data.values(), axis=1)

    # Save the new_df to a new excel file
    new_df.to_excel(save_to, index=False)

if __name__ == "__main__":
    main(PATH, SAVE_TO)