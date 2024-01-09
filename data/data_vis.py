# Data visualizations

# imports
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def time_series_plot(df, x, y, title, xlabel, ylabel, savefig):
    """
    Plot time series data
    """
    plt.figure(figsize=(20, 10))
    plt.plot(df[x], df[y])
    plt.title(title, fontsize=30)
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.savefig(savefig)
    plt.show()

if '__name__' == '__main__':
    # import data as pandas dataframe
    df = pd.read_csv('../data/train.csv')
    # convert date column to datetime
    df['date'] = pd.to_datetime(df['date'])

    # rename columns
    df = df.rename(columns={'date': 'Date', 'store': 'Store', 'item': 'Item', 'sales': 'Sales'})

    # sort by date
    df = df.sort_values(by='date')
    
    # Plot time series
    time_series_plot(df, 'date', 'sales', 'Sales over time', 'Date', 'Sales', 'images/sales_time_series.png')

