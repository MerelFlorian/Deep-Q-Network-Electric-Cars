from data.data_vis import clean_data
import pandas as pd

PATH = "data/train.xlsx"
SAVE_TO = "data/features.xlsx"

def main(path: str) -> None:
    # Load the dataset
    df = clean_data(path)

    # Define columns for featureset
    features = ["3MA", "4MA", "5MA", "6MA", "7MA", "8MA", "10MA", "12MA",
                "3EMA", "4EMA", "5EMA", "6EMA", "7EMA", "8EMA", "10EMA", "12EMA",
                "Season", "Weekend"]
    
    # Create an empty DataFrame with these columns
    new_df = pd.DataFrame(columns=features)

    # Perform your calculations or manipulations here
    # For example, let's create dummy data based on the original df
    new_data = {
        'Feature1': df['Column1'] * 2,  # Assuming 'Column1' exists in your original df
        'Feature2': df['Column2'] + 5,  # Similarly, assuming 'Column2' exists
        'Feature3': len(df['Column3'])  # And 'Column3'
    }

if __name__ == "__main__":
    main(PATH, SAVE_TO)