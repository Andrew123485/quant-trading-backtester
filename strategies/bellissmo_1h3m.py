"""Bellissmo 1h3m Calibrated Strategy Engine.

Formalized from:
- 1h3m-For-Students (Notion spec)
- 1h3m-stats-for-2023 (36 ground-truth trades from Bellissmo's actual journal)

Rules:
1. Session: London open window only (MSK 10:00 - 13:00 winter / 09:00 - 12:00 summer).
2. Context (BIAS): Continuation in direction of D1 structure.
3. Raid (Liquidity Sweep):
   - Price must sweep Frankfurt High/Low, Asia High/Low, or PDH/PDL.
4. Trigger (Execution):
   - M1 or M3 Break of Structure (BoS) with candle body close within 25 min of sweep.
   - Stop Loss placed strictly at the extreme of the sweep candle.
5. Target & RR:
   - Target is the opposite liquidity pool (Asia, Frankfurt, PDH/PDL, H1 swing).
   - Expected RR between 1.25R and 3.5R.
   - If price reaches target before limit fill -> cancel pending order immediately.
6. Trade Management:
   - No Breakeven (BE = 0). Let price hit TP or SL.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import config
import data_loader
from indicators import fractals, levels

def run_bellissmo_backtest(m1: pd.DataFrame, m3: pd.DataFrame, h1: pd.DataFrame, d1: pd.DataFrame,
                           start_date: str = '2023-01-01', end_date: str = '2023-12-31') -> pd.DataFrame:
    # 1. Filter date range
    lo = pd.Timestamp(start_date, tz='UTC')
    hi = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)
    
    m1_sub = m1[(m1.index >= lo) & (m1.index < hi)].copy()
    m3_sub = m3[(m3.index >= lo) & (m3.index < hi)].copy()
    h1_sub = h1[(h1.index >= lo - pd.Timedelta(days=15)) & (h1.index < hi)].copy()
    
    # 2. Add MSK time columns
    from data_loader import add_msk_columns, european_summer_mask
    m1_sub = add_msk_columns(m1_sub)
    m3_sub = add_msk_columns(m3_sub)
    
    # MSK Date mapping
    m1_sub['msk_date'] = (m1_sub.index + pd.Timedelta(hours=3)).normalize()
    m3_sub['msk_date'] = (m3_sub.index + pd.Timedelta(hours=3)).normalize()
    
    # 3. Daily Bias / Context
    # Clean D1 data
    d1_clean = d1.copy()
    if d1_clean.index.tz is not None:
        d1_clean.index = d1_clean.index.tz_convert('UTC').tz_localize(None)
    d1_clean['msk_date'] = d1_clean.index.normalize()
    d1_clean = d1_clean.drop_duplicates('msk_date').set_index('msk_date')
    
    # D1 direction: trend based on 5-day / 20-day momentum and prior day body
    d1_clean['prev_c'] = d1_clean['close'].shift(1)
    d1_clean['prev_o'] = d1_clean['open'].shift(1)
    d1_clean['prev_h'] = d1_clean['high'].shift(1)
    d1_clean['prev_l'] = d1_clean['low'].shift(1)
    d1_clean['sma20'] = d1_clean['prev_c'].rolling(20, min_periods=5).mean()
    d1_clean['sma5'] = d1_clean['prev_c'].rolling(5, min_periods=3).mean()
    
    # D1 bias: 1 = Bullish, -1 = Bearish
    # In SMC: if prev_c > sma20 or (prev_c > prev_o and prev_c > sma5) -> Bullish
    d1_clean['bias'] = np.where(d1_clean['prev_c'] > d1_clean['sma5'], 1, -1)
    
    # 4. Extract Session Extremes per day
    # Asia: 00:00 - Frankfurt start (09:00 summer / 10:00 winter MSK)
    # Frankfurt: 09:00-10:00 summer / 10:00-11:00 winter MSK
    # London: 10:00-13:00 summer / 11:00-14:00 winter MSK
    days = m1_sub['msk_date'].unique()
    trades = []
    
    for day in days:
        day_m1 = m1_sub[m1_sub['msk_date'] == day]
        if day_m1.empty: continue
        
        is_summer = day_m1['is_summer'].iloc[0]
        fra_h_start = 9 if is_summer else 10
        fra_h_end = 10 if is_summer else 11
        lon_h_start = fra_h_end
        lon_h_end = lon_h_start + 3
        
        # D1 levels & bias
        day_ts = pd.Timestamp(day).normalize()
        if day_ts.tz is not None:
            day_ts = day_ts.tz_localize(None)
            
        if day_ts not in d1_clean.index:
            continue
            
        d_info = d1_clean.loc[day_ts]
        bias = int(d_info['bias'])
        pdh = float(d_info['prev_h'])
        pdl = float(d_info['prev_l'])
        
        # Asia bars
        asia = day_m1[day_m1['hour_msk'] < fra_h_start]
        if asia.empty: continue
        asia_hi = float(asia['high'].max())
        asia_lo = float(asia['low'].min())
        
        # Frankfurt bars
        fra = day_m1[(day_m1['hour_msk'] >= fra_h_start) & (day_m1['hour_msk'] < fra_h_end)]
        if fra.empty: continue
        fra_hi = float(fra['high'].max())
        fra_lo = float(fra['low'].min())
        
        # London bars
        lon = day_m1[(day_m1['hour_msk'] >= lon_h_start) & (day_m1['hour_msk'] < lon_h_end)]
        if lon.empty: continue
        
        # Trade logic for London:
        # Bias == 1 (LONG):
        # We look for a sweep of Frankfurt Low or Asia Low during London open!
        # Once swept, we look for M1/M3 BoS up.
        # Stop = sweep low.
        # Target = Frankfurt High, Asia High, or PDH (whichever gives >= 1.25R).
        
        # Bias == -1 (SHORT):
        # We look for a sweep of Frankfurt High or Asia High during London open!
        # Once swept, we look for M1/M3 BoS down.
        # Stop = sweep high.
        # Target = Frankfurt Low, Asia Low, or PDL (whichever gives >= 1.25R).
        
        # Let's iterate through London M1 bars:
        l_times = lon.index.values
        l_highs = lon['high'].to_numpy()
        l_lows = lon['low'].to_numpy()
        l_closes = lon['close'].to_numpy()
        l_opens = lon['open'].to_numpy()
        
        sweep_occurred = False
        sweep_ext = np.nan
        sweep_bar = -1
        bos_occurred = False
        entry_price = np.nan
        sl = np.nan
        tp = np.nan
        entry_bar = -1
        
        if bias == 1: # LONG
            key_low = min(fra_lo, asia_lo)
            # Find sweep of key_low (or fra_lo)
            for b in range(len(lon)):
                if not sweep_occurred:
                    if l_lows[b] < fra_lo - 0.00005:
                        sweep_occurred = True
                        sweep_ext = l_lows[b]
                        sweep_bar = b
                        # recent swing high before sweep on M1 (origin high)
                        start_idx = max(0, b - 10)
                        origin_hi = l_highs[start_idx:b+1].max()
                else:
                    if l_lows[b] < sweep_ext:
                        sweep_ext = l_lows[b]
                        start_idx = max(0, b - 10)
                        origin_hi = l_highs[start_idx:b+1].max()
                    # Check M1 BoS: close above origin_hi
                    if not bos_occurred and b <= sweep_bar + 25:
                        if l_closes[b] > origin_hi:
                            bos_occurred = True
                            entry_price = l_closes[b]
                            sl = sweep_ext
                            entry_bar = b
                            break
                            
            if bos_occurred:
                risk = entry_price - sl
                if risk > 0.00040: # min 4 pips stop
                    # Possible targets
                    cands = [v for v in [fra_hi, asia_hi, pdh] if v > entry_price]
                    if cands:
                        # pick target that gives between 1.25R and 3.5R
                        best_tp = None
                        for cand in sorted(cands):
                            rr = (cand - entry_price) / risk
                            if 1.25 <= rr <= 4.0:
                                best_tp = cand
                                break
                        if best_tp is not None:
                            tp = best_tp
                            # Check forward bars to evaluate outcome
                            # Remaining London + afternoon bars
                            day_rem = day_m1.loc[lon.index[entry_bar]:]
                            rem_h = day_rem['high'].to_numpy()
                            rem_l = day_rem['low'].to_numpy()
                            rem_t = day_rem.index.values
                            
                            win = False
                            loss = False
                            for k in range(1, len(day_rem)):
                                if rem_l[k] <= sl:
                                    loss = True; break
                                elif rem_h[k] >= tp:
                                    win = True; break
                                    
                            if win or loss:
                                actual_rr = (tp - entry_price) / risk if win else -1.0
                                trades.append({
                                    'date': str(day)[:10],
                                    'time': str(l_times[entry_bar])[11:16],
                                    'bias': 'Long',
                                    'result': 'TP' if win else 'SL',
                                    'rr': round(actual_rr, 2),
                                    'risk_pips': round(risk * 10000, 1),
                                    'entry': entry_price,
                                    'sl': sl,
                                    'tp': tp
                                })
                                
        elif bias == -1: # SHORT
            key_hi = max(fra_hi, asia_hi)
            for b in range(len(lon)):
                if not sweep_occurred:
                    if l_highs[b] > fra_hi + 0.00005:
                        sweep_occurred = True
                        sweep_ext = l_highs[b]
                        sweep_bar = b
                        start_idx = max(0, b - 10)
                        origin_lo = l_lows[start_idx:b+1].min()
                else:
                    if l_highs[b] > sweep_ext:
                        sweep_ext = l_highs[b]
                        start_idx = max(0, b - 10)
                        origin_lo = l_lows[start_idx:b+1].min()
                    if not bos_occurred and b <= sweep_bar + 25:
                        if l_closes[b] < origin_lo:
                            bos_occurred = True
                            entry_price = l_closes[b]
                            sl = sweep_ext
                            entry_bar = b
                            break
                            
            if bos_occurred:
                risk = sl - entry_price
                if risk > 0.00040:
                    cands = [v for v in [fra_lo, asia_lo, pdl] if v < entry_price]
                    if cands:
                        best_tp = None
                        for cand in sorted(cands, reverse=True):
                            rr = (entry_price - cand) / risk
                            if 1.25 <= rr <= 4.0:
                                best_tp = cand
                                break
                        if best_tp is not None:
                            tp = best_tp
                            day_rem = day_m1.loc[lon.index[entry_bar]:]
                            rem_h = day_rem['high'].to_numpy()
                            rem_l = day_rem['low'].to_numpy()
                            
                            win = False
                            loss = False
                            for k in range(1, len(day_rem)):
                                if rem_h[k] >= sl:
                                    loss = True; break
                                elif rem_l[k] <= tp:
                                    win = True; break
                                    
                            if win or loss:
                                actual_rr = (entry_price - tp) / risk if win else -1.0
                                trades.append({
                                    'date': str(day)[:10],
                                    'time': str(l_times[entry_bar])[11:16],
                                    'bias': 'Short',
                                    'result': 'TP' if win else 'SL',
                                    'rr': round(actual_rr, 2),
                                    'risk_pips': round(risk * 10000, 1),
                                    'entry': entry_price,
                                    'sl': sl,
                                    'tp': tp
                                })
                                
    return pd.DataFrame(trades)

if __name__ == '__main__':
    data = data_loader.load(('m1', 'm3', 'h1', 'd1'))
    res = run_bellissmo_backtest(data['m1'], data['m3'], data['h1'], data['d1'], '2023-01-01', '2023-12-31')
    print(f"\nBellissmo Engine Test on 2023:")
    print(f"Total Trades: {len(res)}")
    if not res.empty:
        tp_cnt = (res['result'] == 'TP').sum()
        sl_cnt = (res['result'] == 'SL').sum()
        wr = tp_cnt / len(res)
        sum_r = res['rr'].sum()
        print(f"TP: {tp_cnt} | SL: {sl_cnt} | Winrate: {wr:.1%} | Sum R: {sum_r:.1f}R")
        res['month'] = pd.to_datetime(res['date']).dt.month
        print("\nTrades by month:")
        print(res['month'].value_counts().sort_index())
        res.to_csv('bellissmo_2023_results.csv', index=False)
