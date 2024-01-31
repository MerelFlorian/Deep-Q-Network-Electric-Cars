# data_vis.py 
# Data cleaning and visualizations

import pandas as pd
import matplotlib.pyplot as plt
import mplfinance as mpf
from typing import List
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.colors import Normalize, LinearSegmentedColormap
import datetime 
from matplotlib.patches import Patch
import scipy.stats as stats
import scikit_posthocs as sp
import numpy as np


# Functions
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

def candlestick_format(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Convert the DataFrame to candlestick chart format for different time frames.

    Args:
        df (pd.DataFrame): DataFrame with rows for each day and columns for each hour (H1, H2, ..., H24).
        timeframe (str): Time frame for the candlestick chart ('hourly', 'daily', 'weekly', 'monthly').

    Returns:
        pd.DataFrame: DataFrame in the specified time frame candlestick chart format.
    """
    if timeframe not in ['hourly', 'daily', 'weekly', 'monthly']:
        raise ValueError("Invalid timeframe. Choose from 'hourly', 'daily', 'weekly', 'monthly'.")

    # Reshape the DataFrame to have a row for each hour
    hours = []
    for _, row in df.iterrows():
        date = row['date']
        for i in range(1, 25):
            hour = {'datetime': pd.Timestamp(date) + pd.Timedelta(hours=i-1),
                    'value': row[f'H{i}']}
            hours.append(hour)
    reshaped_df = pd.DataFrame(hours)
    reshaped_df.set_index('datetime', inplace=True)

    # Resampling based on the chosen timeframe
    resample_map = {
        'hourly': 'H',
        'daily': 'D',
        'weekly': 'W',
        'monthly': 'M'
    }
    resampled_df = reshaped_df['value'].resample(resample_map[timeframe]).ohlc()
    return resampled_df

def calculate_ema(ohlc: pd.DataFrame, span: int) -> pd.Series:
    """Calculate the Exponential Moving Average for a given span using the 'Open' price

    Args:
        ohlc (pd.DataFrame): DataFrame in OHLC format
        span (int): Span for EMA calculation

    Returns:
        pd.Series: EMA series
    """
    return ohlc['open'].ewm(span=span, adjust=False).mean()

def candlestick(ohlcs: list, descriptions: list) -> None:
    """
    Plot candlestick charts of prices with shared x-axes for different dataframes.

    Args:
        ohlcs (list of pd.DataFrame): List of DataFrames in OHLC format
        descriptions (list of str): List of short descriptions for each chart
    """

    # Create a figure and a list of axes
    fig, axes = plt.subplots(len(ohlcs), 1, figsize=(10, 5 * len(ohlcs)), sharex=True)

    # If only one OHLC DataFrame is provided, wrap it in a list
    if len(ohlcs) == 1:
        ohlcs = [ohlcs]
        axes = [axes]

    for ohlc, ax, description in zip(ohlcs, axes, descriptions):
        # Ensure the index is a DatetimeIndex
        ohlc.index = pd.DatetimeIndex(ohlc.index)

        # Plot candlestick chart on the current axis
        mpf.plot(ohlc, type='candle', style='charles', ax=ax)
        ax.set_title(f'{description}')

    # Adjust layout and save the figure
    plt.tight_layout()
    plt.rcParams.update({'font.size': 25})  # You can adjust the size as needed
    plt.savefig('../images/combined_candlestick_charts.png')

def action_to_color(action)-> None:
    """
    Maps the action value to a color.
    Positive actions (buying) are mapped to red, negative (selling) to blue, and zero (doing nothing) to gray.
    """
    # Negative action (buying)
    if action > 0:
        return (1, 0, 0, 1)  
    # Positive action (selling)
    elif action < 0:
        return (0, 0, 1, 1)  
    # Zero action (doing nothing)
    else:
        return (0.5, 0.5, 0.5, 1)  

def visualize_bat(df: pd.DataFrame, algorithm: str) -> None:
    """
    Visualize the battery level over time with colored action indicators and availability line.

    Args:
        df (pd.DataFrame): DataFrame containing the data
        algorithm (str): Algorithm used
    """
    # Change list to pandas dataframe
    df = pd.DataFrame(df)

    # Calculate the total number of rows in the DataFrame
    total_rows = len(df)

    # Ensure there are at least 72 rows available for selection
    if total_rows >= 72:
        # Generate a random starting index within the valid range
        random_start_index = np.random.randint(0, total_rows - 71)  # Ensure at least 72 rows are remaining
        
        # Select the contiguous subset of 72 rows
        df = df.iloc[random_start_index:random_start_index + 72]
        
        # Now you can use random_72_rows_df for plotting or further analysis
    else:
        print("DataFrame doesn't have enough rows for selection.")

    # Combine date and hour to datetime and add 30 mins to each hour
    df['date'] = df['date'] + pd.to_timedelta(df['hour'] - 1, unit='H') + pd.to_timedelta(30, unit='m')
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
    plt.savefig(f'images/price_battery_level_{algorithm}{i}_new.png')

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
    plt.savefig('images/cumulative_reward{}.png')

def hourly_patterns(df: pd.DataFrame)-> None:
    """Plot hourly patterns for each year

    Args:
        df (pd.Dataframe): DataFrame with hourly data.
    """ 
    # Convert the 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Select only the hourly columns and convert them to numeric values
    hourly_data = df.iloc[:, 1:25].apply(pd.to_numeric, errors='coerce')

    # Set up the figure
    plt.figure(figsize=(20, 10))

    # Define offsets so that the boxplots for each year don't directly overlap
    offsets = [-0.2, 0, 0.2]
    colors = ['skyblue', 'limegreen', 'salmon']  # Colors for different years
    years = [2007, 2008, 2009]

    plt.rcParams.update({'font.size': 17})  # You can adjust the size as needed

    # Create boxplots for each hour and year
    for hour in range(1, 25):
        for i, y in enumerate(years):
            # Filter by year and select the current hour
            hourly_prices = df[df['date'].dt.year == y].iloc[:, hour]

            # Create a boxplot for this hour and year
            bp = plt.boxplot(hourly_prices, positions=[hour + offsets[i]], widths=0.15, patch_artist=True,
                            boxprops=dict(facecolor=colors[i]), medianprops=dict(color="black"), showfliers=False)

    # Title and labels
    plt.title('Hourly Electricity Price Distribution by Year (2007-2009)')
    plt.xlabel('Hour of the Day')
    plt.ylabel('Price (Euros/MWh)')
    plt.xticks(range(1, 25), [f'H{hour}' for hour in range(1, 25)])
    plt.grid(True)

    # Custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=f'Year {y}') for i, y in enumerate(years)]
    plt.legend(handles=legend_elements, title="Year")

    # Save the plot
    plt.savefig('../images/combined_years_hourly_stats_adjusted.png')

def daily_patterns(df: pd.DataFrame) -> None:
    """Plot daily patterns for each year

    Args:
        df (pd.DataFrame): DataFrame with hourly data.
    """
    # Convert the 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Calculate the mean price for each date
    df['mean_daily_price'] = df.iloc[:, 1:25].mean(axis=1)

    # Add column with monday-sunday
    df['day_of_week'] = df['date'].dt.day_name()

    # Define the correct order and colors for the days of the week
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    colors = ['skyblue', 'limegreen', 'salmon']  # Colors for different years
    years = [2007, 2008, 2009]

    plt.rcParams.update({'font.size': 17})  # You can adjust the size as needed

    # Set up the figure
    plt.figure(figsize=(20, 10))

    # Define offsets for each year to avoid overlapping
    offsets = [-0.2, 0, 0.2]

    # Loop for each year and each day
    for i, year in enumerate(years):
        df_year = df[df['date'].dt.year == year]

        # Convert 'day_of_week' to a categorical type with the specified order
        df_year['day_of_week'] = pd.Categorical(df_year['day_of_week'], categories=days_order, ordered=True)

        # Create boxplots
        for day_index, day in enumerate(days_order):
            # Filter data for the specific day of the week
            daily_prices = df_year[df_year['day_of_week'] == day]['mean_daily_price']

            # Create a boxplot for this day and year
            plt.boxplot(daily_prices, positions=[day_index + 1 + offsets[i]], widths=0.15, patch_artist=True,
                        boxprops=dict(facecolor=colors[i]), medianprops=dict(color="black"), showfliers=False)

    # Title and labels
    plt.title('Daily Electricity Price Distribution by Year (2007-2009)')
    plt.xlabel('Day of the Week')
    plt.ylabel('Price (Euro/MWh)')
    plt.xticks(range(1, 8), days_order)
    plt.grid(True)

    # Custom legend
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=f'Year {year}') for i, year in enumerate(years)]
    plt.legend(handles=legend_elements, title="Year")

    # Save the plot
    plt.savefig('../images/daily_price_distribution_years.png')

def monthly_patterns(df: pd.DataFrame) -> None:
    """Plot monthly patterns for each year

    Args:
        df (pd.DataFrame): DataFrame with hourly data.
    """
    # Convert the 'date' column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # Calculate the mean price for each date
    df['mean_daily_price'] = df.iloc[:, 1:25].mean(axis=1)

    # Add column with month
    df['month'] = df['date'].dt.month_name()

    # Define the correct order and colors for the months
    months_order = ['January', 'February', 'March', 'April', 'May', 'June', 
                    'July', 'August', 'September', 'October', 'November', 'December']
    colors = ['skyblue', 'limegreen', 'salmon']  # Colors for different years
    years = [2007, 2008, 2009]

    plt.rcParams.update({'font.size': 17})  # You can adjust the size as needed

    # Set up the figure
    plt.figure(figsize=(20, 10))

    # Define offsets for each year to avoid overlapping
    offsets = [-0.2, 0, 0.2]

    # Loop for each year and each month
    for i, year in enumerate(years):
        df_year = df[df['date'].dt.year == year]

        # Convert 'month' to a categorical type with the specified order
        df_year['month'] = pd.Categorical(df_year['month'], categories=months_order, ordered=True)

        # Create boxplots
        for month_index, month in enumerate(months_order):
            # Filter data for the specific month
            monthly_prices = df_year[df_year['month'] == month]['mean_daily_price']

            # Create a boxplot for this month and year
            plt.boxplot(monthly_prices, positions=[month_index + 1 + offsets[i]], widths=0.15, patch_artist=True,
                        boxprops=dict(facecolor=colors[i]), medianprops=dict(color="black"), showfliers=False)
            
    print(df_year.head())

    # Title and labels
    plt.title('Monthly Electricity Price Distribution by Year (2007-2009)')
    plt.xlabel('Month')
    plt.ylabel('Price (Euro/MWh)')
    plt.xticks(range(1, 13), months_order)
    plt.grid(True)

    # Custom legend
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=f'Year {year}') for i, year in enumerate(years)]
    plt.legend(handles=legend_elements, title="Year")

    # Save the plot
    plt.savefig('../images/monthly_price_distribution_years.png')

def stats_season(df: pd.DataFrame)->None:
    """Test hypothesis for hourly data (ANOVA or Kruskal wallis test)

    Args:
        df (pd.DataFrame): DataFrame with hourly data.
    """
    # Convert the 'date' column to datetime
    df['date'] = pd.to_datetime(df['datetime'])

    # Add season column
    df['season'] = df['date'].dt.month.apply(lambda x: 'winter' if x in [12, 1, 2] else 'spring' if x in [3, 4, 5] else 'summer' if x in [6, 7, 8] else 'fall')

    # Kruskal wallis test
    print(stats.kruskal(df[df['season'] == 'winter']['Value'],df[df['season'] == 'fall']['Value'],df[df['season'] == 'summer']['Value'], df[df['season'] == 'spring']['Value']))

    # Post hoc test
    # Perform Dunn's test for post hoc analysis
    print(sp.posthoc_dunn(df, val_col='Value', group_col='season', p_adjust='bonferroni'))

    # Statistical table with median, Q1, Q3, min and max
    print(df.groupby('season')['Value'].describe())
    print(df.groupby('season')['Value'].median())

def table_summary(df: pd.DataFrame) -> None:
    """Saves a table with summary statistics for each year and the entire dataset

    Args:
        df (pd.DataFrame): DataFrame with hourly data in long format.
    """
    # Convert the 'date' column to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])

    # Create a DataFrame for summary statistics
    df_table = pd.DataFrame()

    # Add summary statistics for each year
    for year in [2007, 2008, 2009]:
        df_year = df[df['datetime'].dt.year == year]
        df_table[year] = [
            round(df_year['Value'].mean(), 2),
            round(df_year['Value'].median(), 2),
            round(df_year['Value'].min(), 2),
            round(df_year['Value'].max(), 2),
            round(df_year['Value'].quantile(0.25), 2),
            round(df_year['Value'].quantile(0.75), 2),
            df_year['Value'].isna().sum()
        ]

    # Add summary statistics for the entire dataset
    df_table['Overall'] = [
        round(df['Value'].mean(), 2),
        round(df['Value'].median(), 2),
        round(df['Value'].min(), 2),
        round(df['Value'].max(), 2),
        round(df['Value'].quantile(0.25), 2),
        round(df['Value'].quantile(0.75), 2),
        df['Value'].isna().sum()
    ]
    # Give percentage of values above 200
    print((df[df['Value'] > 200]['Value'].count()/df['Value'].count())*100)

    # Clip dataset to 200 and add summary statistics
    df_clip = df[df['Value'] <= 200]

    df_table['Overall (clip 200)'] = [
        round(df_clip['Value'].mean(), 2),
        round(df_clip['Value'].median(), 2),
        round(df_clip['Value'].min(), 2),
        round(df_clip['Value'].max(), 2),
        round(df_clip['Value'].quantile(0.25), 2),
        round(df_clip['Value'].quantile(0.75), 2),
        df_clip['Value'].isna().sum()
    ]
    
    # Set the index for better readability
    df_table.index = ['Mean', 'Median', 'Min', 'Max', 'Q1', 'Q3', 'NA Count']

    # Print the table
    print(df_table)

    # Save the table to a csv file
    df_table.to_csv('../data/summary_statistics.csv')

if __name__ == "__main__":
    # Load train and val dataset
    df_train = clean_data('../data/train.xlsx')
    df_val = clean_data('../data/validate.xlsx')

    # Concatinate them
    df = pd.concat([df_train, df_val])
    # List of years
    years = df['date'].dt.year.unique()

    # Create a new DataFrame to store mean values for each year
    mean_table = pd.DataFrame()

    # Loop over years
    for year in years:
        # Filter df for year
        df_year = df[df['date'].dt.year == year]

        # Remove date row
        df_year = df_year.drop('date', axis=1)

        # Calculate mean for each column
        yearly_means = df_year.mean()

        # Add mean values as a column to mean_table
        mean_table[year] = yearly_means.round(2)

    # Save mean_table to CSV
    mean_table.to_csv('../data/yearly_hourly_means.csv')


