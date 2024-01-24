
from data_vis import candlestick, candlestick_format, hourly_patterns, daily_patterns, monthly_patterns, clean_data, table_summary, long_format, stats_season

if __name__ == "__main__":
    # Clean the data
    df = clean_data('train.xlsx')

    # Stats for the weekend
    stats_season(long_format(df))

    # Table summary
    table_summary(long_format(df))
    
    # Add columns to clip to 200
    columns_to_clip = [f'H{i}' for i in range(1, 25)]  # List of columns H1, H2, ..., H24

    # Applying clip to each of these columns
    for col in columns_to_clip:
        df[col] = df[col].clip(upper=200)
 
    # Plot the patterns
    hourly_patterns(df)
    daily_patterns(df)
    monthly_patterns(df)

    # Plot weekly candlesticks
    ohlc = []
    # Break data in 3 years
    for y in [2007,2008,2009]:
        df_new = df[df['date'].dt.year == y]
        ohlc.append(candlestick_format(df_new, 'weekly'))
    candlestick(ohlc, ['2007', '2008', '2009'])