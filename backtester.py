import pandas as pd
import numpy as np
import pytz

def load_data():
    df_h1 = pd.read_csv("EURUSD_h1.csv")
    df_m3 = pd.read_csv("EURUSD_m3.csv")
    
    # MT5 time is broker time, but wait, copy_rates returns UNIX timestamp which is UTC.
    # Let's verify: if it's UNIX timestamp, unit='s' makes it UTC.
    df_h1['time_utc'] = pd.to_datetime(df_h1['time'], unit='s', utc=True)
    df_m3['time_utc'] = pd.to_datetime(df_m3['time'], unit='s', utc=True)
    
    # Convert to MSK
    msk_tz = pytz.timezone('Europe/Moscow')
    berlin_tz = pytz.timezone('Europe/Berlin')
    
    df_h1['time_msk'] = df_h1['time_utc'].dt.tz_convert(msk_tz)
    df_m3['time_msk'] = df_m3['time_utc'].dt.tz_convert(msk_tz)
    
    # Determine Summer/Winter based on Europe/Berlin
    df_m3['is_summer'] = df_m3['time_utc'].dt.tz_convert(berlin_tz).apply(lambda x: x.dst().total_seconds() != 0)
    
    # Trading window MSK
    # Summer: 9:00 - 12:00
    # Winter: 10:00 - 13:00
    df_m3['hour_msk'] = df_m3['time_msk'].dt.hour
    df_m3['minute_msk'] = df_m3['time_msk'].dt.minute
    
    df_m3['in_trading_window'] = False
    
    # Summer condition
    summer_mask = df_m3['is_summer']
    df_m3.loc[summer_mask & (df_m3['hour_msk'] >= 9) & (df_m3['hour_msk'] < 12), 'in_trading_window'] = True
    
    # Winter condition
    winter_mask = ~df_m3['is_summer']
    df_m3.loc[winter_mask & (df_m3['hour_msk'] >= 10) & (df_m3['hour_msk'] < 13), 'in_trading_window'] = True
    
    df_h1.set_index('time_msk', inplace=True)
    df_m3.set_index('time_msk', inplace=True)
    
    return df_h1, df_m3

def calculate_pdh_pdl(df_h1):
    # Daily logic based on MSK date
    # Technically, trading day changes at NY close, but for PDH/PDL usually it's the daily high/low
    # Let's resample by day
    daily = df_h1.resample('D').agg({'high': 'max', 'low': 'min'}).dropna()
    daily['PDH'] = daily['high'].shift(1)
    daily['PDL'] = daily['low'].shift(1)
    
    df_h1['date'] = df_h1.index.normalize()
    daily_shifted = daily[['PDH', 'PDL']]
    
    df_h1 = df_h1.join(daily_shifted, on='date')
    return df_h1

def calculate_fractals(df_h1, period=2):
    df_h1['fractal_high'] = False
    df_h1['fractal_low'] = False
    df_h1['fractal_high_val'] = np.nan
    df_h1['fractal_low_val'] = np.nan
    
    window = period * 2 + 1
    
    rolling_max = df_h1['high'].rolling(window=window, center=True).max()
    rolling_min = df_h1['low'].rolling(window=window, center=True).min()
    
    is_fractal_high = (df_h1['high'] == rolling_max)
    is_fractal_low = (df_h1['low'] == rolling_min)
    
    df_h1.loc[is_fractal_high, 'fractal_high'] = True
    df_h1.loc[is_fractal_low, 'fractal_low'] = True
    df_h1.loc[is_fractal_high, 'fractal_high_val'] = df_h1['high']
    df_h1.loc[is_fractal_low, 'fractal_low_val'] = df_h1['low']
    
    df_h1['last_fractal_high'] = df_h1['fractal_high_val'].ffill()
    df_h1['last_fractal_low'] = df_h1['fractal_low_val'].ffill()
    
    return df_h1

def main():
    df_h1, df_m3 = load_data()
    print("M3 sample with trading window:")
    print(df_m3[['close', 'is_summer', 'hour_msk', 'in_trading_window']].head(10))
    
    df_h1 = calculate_pdh_pdl(df_h1)
    df_h1 = calculate_fractals(df_h1, period=2)
    
    print("H1 sample with fractals:")
    print(df_h1[['high', 'low', 'PDH', 'PDL', 'fractal_high_val']].dropna(subset=['fractal_high_val']).head(5))

if __name__ == "__main__":
    main()
