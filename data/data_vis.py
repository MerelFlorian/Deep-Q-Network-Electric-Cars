# Data cleaning and visualizations

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import List
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
from collections import defaultdict
import matplotlib.cm as cm
import datetime 
import copy
import itertools


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

    # Formatting
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

def candlestick_hourly(ohlc: pd.DataFrame, description: str, ema_spans: List[int]) -> None:
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
    mpf.plot(ohlc, type='candle', style='charles', title=f'Hourly Candlestick Chart {description}', \
             mav=tuple(ema_spans), savefig=f'../images/candlestick_hourly_{description}.png')

def action_to_color(action)-> None:
    """
    Maps the action value to a color.
    Positive actions (buying) are mapped to red, negative (selling) to blue, and zero (doing nothing) to gray.
    """
    if action < 0:
        return (1, 0, 0, 1)  # Solid red for buying
    elif action > 0:
        return (0, 0, 1, 1)  # Solid blue for selling
    else:
        return (0.5, 0.5, 0.5, 1)  # Gray for doing nothing

def visualize_bat(df: pd.DataFrame, algorithm: str) -> None:
    """
    Visualize the battery level over time with colored action indicators and availability line.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        algorithm (str): Algorithm used
    """
    # Convert defaultdict to DataFrame and limit the data to the first 48 rows (2 days)
    df = pd.DataFrame(df).head(72)

    # Ensure the 'date' column is in datetime format
    df['date'] = pd.to_datetime(df['date'])

    # Create a custom colormap
    cmap = LinearSegmentedColormap.from_list("action_colormap", [(1, 0, 0), (0.5, 0.5, 0.5), (0, 0, 1)])
    norm = Normalize(vmin=df['action'].min(), vmax=df['action'].max())

    # Creating a new figure and axes
    fig, ax1 = plt.subplots(figsize=(10, 6))
    plt.subplots_adjust(right=0.85)  # Adjust the right margin

    # Plot price line
    ax1.plot(df['date'], df['price'], label='Price', color='black', alpha=0.5)

    # Plot colored dots for actions on top of the price line
    colors = [action_to_color(a) for a in df['action']]
    ax1.scatter(df['date'], df['price'], color=colors)

    # Add a solid block of gray for the period between 8:00 AM and 6:00 PM where battery is not available
    for i in range(len(df['date'])):
        date = df['date'].iloc[i]
        start_block = datetime.datetime(date.year, date.month, date.day, 8, 0, 0)
        end_block = datetime.datetime(date.year, date.month, date.day, 18, 0, 0)
        if not df['availability'].iloc[i] and start_block <= date < end_block:
            ax1.fill_betweenx([min(df['price']), max(df['price'])], start_block, end_block, color='gray', alpha=0.5)

    # Plot a horizontal line for availability
    for i in range(1, len(df['date'])):
        color = 'green' if df['availability'].iloc[i] else 'black'
        ax1.plot([df['date'].iloc[i-1], df['date'].iloc[i]], [min(df['price']) - 1, min(df['price']) - 1], color=color, lw=5)

    ax1.set_xlabel('Time (Hours)')
    ax1.set_ylabel('Price (Euros/MWh)', color='black')
    ax1.tick_params('y', colors='black')

    # Formatting the x-axis to display time in hours
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%H'))
    ax1.xaxis.set_major_locator(mdates.HourLocator(interval=2))

    # Adding battery level with secondary axis
    ax2 = ax1.twinx()
    ax2.plot(df['date'], df['battery'], label='Battery Level', color='purple')
    ax2.set_ylabel('Battery Level (kWh)', color='purple')
    ax2.tick_params('y', colors='purple')

    # Create custom legend elements for actions
    legend_elements_actions = [
        Line2D([0], [0], marker='o', color='w', label='Sell',
            markersize=10, markerfacecolor='blue'),
        Line2D([0], [0], marker='o', color='w', label='Buy', 
               markersize=10, markerfacecolor='red'),
        Line2D([0], [0], marker='o', color='w', label='Do nothing',
            markersize=10, markerfacecolor='gray'),
        # Additional legend element for unavailability
        Line2D([0], [0], color='gray', lw=4, label='Unavailable 8 AM-6 PM')
    ]

    # Create custom legend elements for availability
    legend_elements_availability = [
        Line2D([0], [0], color='green', lw=2, label='Available (Green)'),
        Line2D([0], [0], color='black', lw=2, label='Unavailable (Black)')
    ]

    # Combine the legend elements
    combined_legend_elements = legend_elements_actions + legend_elements_availability

    # Add the combined custom legend to one of the axes
    ax1.legend(handles=combined_legend_elements, loc='upper left', fontsize='small')

    # Add title
    # Main title
    plt.title(f'Price and Battery Level Over Time for {algorithm}', color='black')
    plt.suptitle(f'Timespan: [{df["date"].iloc[0].strftime("%Y-%m-%d")} - {df["date"].iloc[-1].strftime("%Y-%m-%d")}]', color='gray', fontsize=10, style='italic')

    plt.tight_layout()
    plt.savefig(f'images/price_battery_level_{algorithm}.png')

def plot_revenue(log_env_ql, log_env_blsh, log_env_ema) -> None:
    """
    Plot the cumulative rewards for each agent
    """
    
    # Convert dates to days since the start for plotting
    days_ql = [(date - log_env_ql['date'][0]).days for date in log_env_ql['date']]
    days_blsh = [(date - log_env_blsh['date'][0]).days for date in log_env_blsh['date']]
    days_ema = [(date - log_env_ema['date'][0]).days for date in log_env_ema['date']]

    # Plot cumulative rewards for each agent
    plt.figure(figsize=(12, 6))

    plt.plot(days_ql, log_env_ql['revenue'], label='QLearningAgent')
    plt.plot(days_blsh, log_env_blsh['revenue'], label='BuyLowSellHigh')
    plt.plot(days_ema, log_env_ema['revenue'], label='EMA')

    plt.xlabel('Days')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Rewards over Days for Baseline Algorithms')
    plt.legend()
    plt.grid(True)
    plt.savefig('images/cumulative_reward.png')


if __name__ == "__main__":
    # Clean the data
    df = clean_data('train_clean.csv')
    # # Add date columns
    # df = add_date_columns(df)
    # # Save the DataFrame to a csv file
    # df.to_csv('data/train_clean.csv', index=False)

    # # Clean the data
    # df = clean_data('data/train.csv')

    # get deepcopy of df for candlestick chart
    df_candle = copy.deepcopy(df)

    # # # Make a plot for each year
    # # for y in [2007,  2008, 2009]:
    # #     # Get candlestick dataset for each year
    # #     df_year = df_candle[df_candle['date'].dt.year == y]
    # #     # Convert dataset to candlestick chart format
    # #     ohlc_year = candlestick_format(df_year)
    # #     # Make dataframe for candlestick chart and save 
    # #     candlestick(ohlc_year, str(y))

    # Select the first week of 2007
    start_date = '2007-02-01'
    end_date = '2007-02-02'
    df_2007_first_week = df_candle[(df_candle['date'] >= start_date) & (df_candle['date'] <= end_date)]

    # Convert dataset to hourly candlestick chart format
    ohlc_hourly = hourly_candlestick_format(df_2007_first_week)

    # Generate a list of all combinations from (2,3) to (11,12)
    options_range = range(2, 13)  # Since the last value in Python's range is exclusive, we use 13
    all_combinations = [[i, j] for i, j in itertools.combinations(options_range, 2)]

    for range in all_combinations:
        # Make dataframe for hourly candlestick chart and save 
        candlestick_hourly(ohlc_hourly, f'first_week_2007_{range}', range)  # Adding 12-hour and 26-hour EMAs

    # # Convert the DataFrame to long format
    # df_long = long_format(df)
    
    # # Save the DataFrame to a csv file
    # # Save the DataFrame to a csv file
    # df_long.to_csv('data/long_format.csv', index=False) 