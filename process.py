from data.data_vis import clean_data
import pandas as pd

PATH = "data/train.xlsx"
SAVE_TO = "data/features.xlsx"

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

    # Add the 'Season' column
    df['Season'] = df['Month'].apply(season)

    # Add a 'Weekend' feature
   
    df['Weekend'] = df['Day'].apply(weekend)

    # Define columns for featureset
    features = ["3MA", "4MA", "5MA", "6MA", "7MA", "8MA", "10MA", "12MA",
                "3EMA", "4EMA", "5EMA", "6EMA", "7EMA", "8EMA", "10EMA", "12EMA",
                "Season", "Weekend"]
    
    # Create an empty DataFrame with these columns
    new_df = pd.DataFrame(columns=features)

    new_data = {
        'Season': df['Season'],
        'Weekend': df["Weekend"]
    }

if __name__ == "__main__":

    main(PATH, SAVE_TO)