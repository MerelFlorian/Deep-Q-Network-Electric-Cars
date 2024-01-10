# Data visualizations

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io

def clean_data(csv_file: str) -> pd.DataFrame:
    """Clean the data. Removes NAN values and removes all columns after 24th column. 
    Removes empty rows. Dates are converted to datetime format. Columns are renamed to H1, H2, H3.

    Args:
        csv_file (str): Path to csv file

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """

    # Import data as pandas dataframe
    df = pd.read_csv('../data/train.csv')
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

def add_date_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add year, month, day columns to DataFrame

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: DataFrame with year, month, day columns
    """
    # Split date column into year, month, day
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day

    return df

def long_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the DataFrame to long format. This is done by melting the DataFrame and adding a timedelta to the dates

    Args:
        df (pd.DataFrame): DataFrame

    Returns:
        pd.DataFrame: Long format DataFrame
    """
    # Melt the DataFrame to get a long format DataFrame with dates and values
    df_long = df.melt(id_vars=['date'], var_name='Hour', value_name='Value')

    # Convert the 'Hour' column to represent the actual hour as a timedelta
    df_long['Hour'] = pd.to_timedelta(df_long['Hour'].str.extract('(\d+)$')[0].astype(int), unit='H')

    # Add the hour timedelta to the dates to get the actual datetime
    df_long['datetime'] = df_long['date'] + df_long['Hour']

    # Drop the now unnecessary columns
    df_long.drop(['date', 'Hour'], axis=1, inplace=True)

    # Sort by datetime to ensure the plot is in order
    df_long.sort_values(by='datetime', inplace=True)

    # Save the DataFrame to a csv file
    df_long.to_csv('../data/long_format.csv', index=False)

    return df_long  

def plot_daily_average_values_per_year(df_long: pd.DataFrame)-> None:
    """Plot the daily average values per year

    Args:
        df_long (pd.DataFrame): Long format DataFrame

    Returns:
        None: Plot is saved to images folder
    """
    # Plotting
    plt.figure(figsize=(15, 5))
    for year in df_long['datetime'].dt.year.unique():
        # Select data for the year
        year_data = df_long[df_long['datetime'].dt.year == year]
        # Group by day and calculate the mean for each day
        daily_mean = year_data.set_index('datetime').resample('D').mean()
        # Plot
        plt.plot(daily_mean.index, daily_mean['Value'], label=str(year))

    plt.xlabel('Datetime')
    plt.ylabel('Average Daily Value')
    plt.title('Daily Average Values Per Year')
    plt.legend()
    plt.savefig('../images/daily_average_values_per_year.png')

if "__name__" == "__main__":
    # Clean the data
    df = clean_data('../data/train.csv')
    # Add date columns
    df = add_date_columns(df)
    # Save the DataFrame to a csv file
    df.to_csv('../data/train_clean.csv', index=False)
    # Convert the DataFrame to long format
    df_long = long_format(df)
    # Save the DataFrame to a csv file
    df_long.to_csv('../data/long_format.csv', index=False)
    # Plot the daily average values per year and save to images folder
    plot_daily_average_values_per_year(df_long)


