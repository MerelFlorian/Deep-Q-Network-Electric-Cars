# Data visualizations

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import io
import mplfinance as mpf
import copy 

def clean_data(csv_file: str) -> pd.DataFrame:
    """Clean the data. Removes NAN values and removes all columns after 24th column. 
    Removes empty rows. Dates are converted to datetime format. Columns are renamed to H1, H2, H3.

    Args:
        csv_file (str): Path to csv file

    Returns:
        pd.DataFrame: Cleaned DataFrame
    """

    # Import data as pandas dataframe
    df = pd.read_csv(csv_file)
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
    df_long.to_csv('long_format.csv', index=False)

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
    plt.savefig('images/daily_average_values_per_year.png')

def candlestick_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the DataFrame to weekly candlestick chart format.

    Args:
        df (pd.DataFrame): DataFrame with hourly data and a 'date' column.

    Returns:
        pd.DataFrame: DataFrame in weekly candlestick chart format.
    """
    # Ensure 'date' is a datetime and set it as index
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)

    # Calculate 'High' and 'Low' for each day before resampling
    df['High'] = df.loc[:, 'H1':'H24'].max(axis=1)
    df['Low'] = df.loc[:, 'H1':'H24'].min(axis=1)

    # Resample the data to weekly frequency
    # 'Open' is the first 'H1' value of the week, 'Close' is the last 'H23' value of the week
    weekly_df = df.resample('W').agg({'H1': 'first', 
                                      'H24': 'last', 
                                      'High': 'max', 
                                      'Low': 'min'})

    # Rename columns to match OHLC convention
    weekly_df.rename(columns={'H1': 'Open', 'H24': 'Close'}, inplace=True)

    return weekly_df[['Open', 'High', 'Low', 'Close']]

def candlestick(ohlc: pd.DataFrame, description) -> None:
    """Plot a candlestick chart of prices.

    Args:
        df (pd.DataFrame): DataFrame
        description (str): Short description of the chart
    """
    # Plot candlestick chart
    mpf.plot(ohlc, type='candle', style='charles', title=f'Weekly Candlestick Chart {description}', savefig=f'images/candlestick{description}.png')

def calculate_ema(ohlc: pd.DataFrame, span: int) -> pd.Series:
    """Calculate the Exponential Moving Average for a given span using the 'Open' price

    Args:
        ohlc (pd.DataFrame): DataFrame in OHLC format
        span (int): Span for EMA calculation

    Returns:
        pd.Series: EMA series
    """
    return ohlc['Open'].ewm(span=span, adjust=False).mean()

def hourly_candlestick_format(df: pd.DataFrame) -> pd.DataFrame:
    """Convert the DataFrame to hourly candlestick chart format.

    Args:
        df (pd.DataFrame): DataFrame with hourly data.

    Returns:
        pd.DataFrame: DataFrame in hourly candlestick chart format.
    """
    # Reshape the DataFrame to have a row for each hour
    hours = []
    for _, row in df.iterrows():
        date = row['date']
        for i in range(1, 25):
            hour = {'datetime': pd.Timestamp(date) + pd.Timedelta(hours=i-1),
                    'Value': row[f'H{i}']}
            hours.append(hour)
    hourly_df = pd.DataFrame(hours)

    # Determine 'Open', 'High', 'Low', 'Close' for each hour
    hourly_df['Open'] = hourly_df['Value']
    hourly_df['High'] = hourly_df['Value']
    hourly_df['Low'] = hourly_df['Value']
    hourly_df['Close'] = hourly_df['Value']

    hourly_df.set_index('datetime', inplace=True)
    
    return hourly_df[['Open', 'High', 'Low', 'Close']]

def candlestick_hourly(ohlc: pd.DataFrame, description: str, ema_spans: list[int]) -> None:
    """Plot an hourly candlestick chart of prices with EMAs.

    Args:
        ohlc (pd.DataFrame): DataFrame in OHLC format
        description (str): Short description of the chart
        ema_spans (list[int]): List of spans for EMA calculation
    """
    # Add EMAs to the DataFrame
    for span in ema_spans:
        ohlc[f'EMA_{span}'] = ohlc['Open'].ewm(span=span, adjust=False).mean()

    # Ensure the index is a DatetimeIndex
    ohlc.index = pd.DatetimeIndex(ohlc.index)

    # Plot candlestick chart
    mpf.plot(ohlc, type='candle', style='charles', title=f'Hourly Candlestick Chart {description}', 
             mav=tuple(ema_spans), savefig=f'images/candlestick_hourly_{description}.png')
import matplotlib.pyplot as plt

def plot_battery_levels(battery_levels, title='Battery Levels', save_path='images/battery_levels.png'):
    """
    Plots the battery levels.

    Parameters:
    battery_levels (list): List of battery levels to plot.
    title (str): Title for the plot.
    """
    plt.figure(figsize=(10, 4))
    plt.plot(battery_levels)
    plt.title(title)
    plt.xlabel('Time Step')
    plt.ylabel('Battery Level')
    plt.grid(True)
    plt.savefig(save_path, bbox_inches='tight')

if __name__ == "__main__":
    # Clean the data
    df = clean_data('data/train.csv')
    # Add date columns
    df = add_date_columns(df)
    # Save the DataFrame to a csv file
    df.to_csv('data/train_clean.csv', index=False)

    # Clean the data
    df = clean_data('data/train.csv')

    # get deepcopy of df for candlestick chart
    df_candle = copy.deepcopy(df)

    # # Make a plot for each year
    # for y in [2007,  2008, 2009]:

    #     # Get candlestick dataset for each year
    #     df_year = df_candle[df_candle['date'].dt.year == y]
    #     print(df_year.head())

    #     # Convert dataset to candlestick chart format
    #     ohlc_year = candlestick_format(df_year)
    #     print(ohlc_year.head())

    #     # Make dataframe for candlestick chart and save 
    #     candlestick(ohlc_year, str(y))

    # Select the first week of 2007
    start_date = '2007-02-01'
    end_date = '2007-02-07'
    df_2007_first_week = df_candle[(df_candle['date'] >= start_date) & (df_candle['date'] <= end_date)]

    # Convert dataset to hourly candlestick chart format
    ohlc_hourly = hourly_candlestick_format(df_2007_first_week)

    # Make dataframe for hourly candlestick chart and save 
    candlestick_hourly(ohlc_hourly, 'first_week_2007', [3, 12])  # Adding 12-hour and 26-hour EMAs

    # # Convert the DataFrame to long format
    # df_long = long_format(df)
    
    # # Save the DataFrame to a csv file
    # df_long.to_csv('data/long_format.csv', index=False)

    # # Plot the daily average values per year and save to images folder
    # plot_daily_average_values_per_year(df_long)