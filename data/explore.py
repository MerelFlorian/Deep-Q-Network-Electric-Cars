
from data_vis import candlestick, candlestick_format, hourly_patterns, daily_patterns, monthly_patterns, clean_data, table_summary, long_format
import pandas as pd

if __name__ == "__main__":
    # Clean the data
    df = clean_data('train.xlsx')

    # Table summary
    table_summary(long_format(df))
    
    # Add columns to clip to 200
    columns_to_clip = [f'H{i}' for i in range(1, 25)]  # List of columns H1, H2, ..., H24

    # Applying clip to each of these columns
    for col in columns_to_clip:
        df[col] = df[col].clip(upper=200)
    
     # # # Make a plot for each year
    # # for y in [2007,  2008, 2009]:
    # #     # Get candlestick dataset for each year
    # #     df_year = df_candle[df_candle['date'].dt.year == y]
    # #     # Convert dataset to candlestick chart format
    # #     ohlc_year = candlestick_format(df_year)
    # #     # Make dataframe for candlestick chart and save 
    # #     candlestick(ohlc_year, str(y))

    # # Select the first two weeks of 2007
    # start_date = '2007-02-01'
    # end_date = '2007-02-14'
    # df_2007_first_week = df_candle[(df_candle['date'] >= start_date) & (df_candle['date'] <= end_date)]

    # frame = "hourly"
    
    # for y in [2007,  2008, 2009]:
    #     df_c = df[df['date'].dt.year == y]
    #     print(df_c.head())
    #     print(frame)
        
    #     # Convert dataset to candlestick chart format
    #     ohlc = candlestick_format(df_c, frame)

    #     # Make dataframe for hourly candlestick chart and save 
    #     candlestick(ohlc, f'{y}_{frame}_clipped')  # Adding 12-hour and 26-hour EMAs
        
    hourly_patterns(df)
    daily_patterns(df)
    monthly_patterns(df)

    ohlc = []
    # Break data in 3 years
    for y in [2007,2008,2009]:
        df_new = df[df['date'].dt.year == y]
        ohlc.append(candlestick_format(df_new, 'weekly'))
    candlestick(ohlc, ['2007', '2008', '2009'])