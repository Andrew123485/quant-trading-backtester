"""Test Method 4 Complete Sniper across all assets with Asia + Frankfurt Sweeps:
- Evaluates EURUSD, GBPUSD, GER40, USDCHF, USDJPY, US500
- 2021 - 2024
- Calculate:
  1. Winrate per year
  2. Total R per year (where 1R = 1% = $100)
  3. Max Drawdown in R and %
  4. Worst Month in R and %
  5. Month by month breakdown
"""
import os, sys, time
import numpy as np
import pandas as pd
import data_loader
from strategies import context_h1

ASSET_SPECS = {
    'EURUSD': {'pkl': 'data_m1.pkl', 'pip': 0.0001, 'min_risk': 0.00045, 'mult': 10000, 'session': 'London'},
    'GBPUSD': {'pkl': 'data_gbpusd_m1.pkl', 'pip': 0.0001, 'min_risk': 0.00055, 'mult': 10000, 'session': 'London'},
    'USDJPY': {'pkl': 'data_usdjpy_m1.pkl', 'pip': 0.01, 'min_risk': 0.06, 'mult': 100, 'session': 'London'},
    'USDCHF': {'pkl': 'data_usdchf_m1.pkl', 'pip': 0.0001, 'min_risk': 0.00045, 'mult': 10000, 'session': 'London'},
    'GER40':  {'pkl': 'data_ger40_m1.pkl', 'pip': 5.0, 'min_risk': 12.0, 'mult': 1, 'session': 'London'},
    'US500':  {'pkl': 'data_us500_m1.pkl', 'pip': 1.0, 'min_risk': 2.5, 'mult': 1, 'session': 'NewYork'},
}

def backtest_asset(sym, spec, start_year=2021, end_year=2024):
    pkl = spec['pkl']
    if not os.path.exists(pkl): return pd.DataFrame()
    m1 = pd.read_pickle(pkl)
    lo = pd.Timestamp(f"{start_year}-01-01", tz='UTC')
    hi = pd.Timestamp(f"{end_year}-12-31 23:59:59", tz='UTC')
    
    m1_sub = m1[(m1.index >= lo - pd.Timedelta(days=20)) & (m1.index <= hi)].copy()
    ohlc = m1_sub[['open', 'high', 'low', 'close']]
    h1 = data_loader._resample(ohlc, '1h')
    d1 = data_loader._resample(ohlc, '1D')
    
    m1_eval = m1[(m1.index >= lo) & (m1.index <= hi)].copy()
    m1_eval = data_loader.add_msk_columns(m1_eval)
    m1_eval['msk_date'] = (m1_eval.index + pd.Timedelta(hours=3)).normalize()
    
    ctx = context_h1.detect(h1, penetration=spec['pip'])
    ctx_closed = ctx[['ctx_dir', 'ctx_sweep']].copy()
    ctx_closed.index = ctx_closed.index + pd.Timedelta(hours=1)
    
    m1r = m1_eval.reset_index()
    h1r = ctx_closed.reset_index()
    m1r = m1r.rename(columns={m1r.columns[0]: 'time'})
    h1r = h1r.rename(columns={h1r.columns[0]: 'time'})
    m1r['time'] = m1r['time'].dt.as_unit('ns')
    h1r['time'] = h1r['time'].dt.as_unit('ns')
    merged = pd.merge_asof(m1r, h1r, on='time', direction='backward').set_index('time')
    
    d1_clean = d1.copy()
    if d1_clean.index.tz is not None:
        d1_clean.index = d1_clean.index.tz_convert('UTC').tz_localize(None)
    d1_clean['msk_date'] = d1_clean.index.normalize()
    d1_dict = d1_clean.drop_duplicates('msk_date').set_index('msk_date')[['high', 'low']].to_dict('index')
    
    trades = []
    pen = spec['pip']
    min_risk = spec['min_risk']
    
    for day, day_bars in merged.groupby('msk_date', sort=False):
        if day_bars.empty: continue
        is_summer = day_bars['is_summer'].iloc[0]
        fra_start = 9 if is_summer else 10
        fra_end = 10 if is_summer else 11
        lon_start = fra_end
        lon_end = lon_start + 2.5
        
        day_ts = pd.Timestamp(day).normalize()
        if day_ts.tz is not None: day_ts = day_ts.tz_localize(None)
        if day_ts not in d1_dict: continue
        d_info = d1_dict[day_ts]
        pdh = float(d_info['high'])
        pdl = float(d_info['low'])
        
        asia = day_bars[day_bars['hour_msk'] < fra_start]
        fra = day_bars[(day_bars['hour_msk'] >= fra_start) & (day_bars['hour_msk'] < fra_end)]
        if asia.empty or fra.empty: continue
        
        asia_hi, asia_lo = float(asia['high'].max()), float(asia['low'].min())
        fra_hi, fra_lo = float(fra['high'].max()), float(fra['low'].min())
        
        if spec['session'] == 'London':
            fra_last_10 = fra[fra['minute_msk'] >= 50]
            if not fra_last_10.empty:
                if fra_last_10['high'].max() >= fra_hi or fra_last_10['low'].min() <= fra_lo:
                    continue
            ws, we = lon_start, lon_end
            sweep_targets_hi = [fra_hi, asia_hi]
            sweep_targets_lo = [fra_lo, asia_lo]
        else: # New York for US500
            ws = 16 if is_summer else 17
            we = ws + 2.5
            lon_full = day_bars[(day_bars['hour_msk'] >= lon_start) & (day_bars['hour_msk'] < ws)]
            lon_hi = float(lon_full['high'].max()) if not lon_full.empty else fra_hi
            lon_lo = float(lon_full['low'].min()) if not lon_full.empty else fra_lo
            sweep_targets_hi = [lon_hi, fra_hi]
            sweep_targets_lo = [lon_lo, fra_lo]
            
        w_bars = day_bars[(day_bars['hour_msk'] >= ws) & (day_bars['hour_msk'] < we)]
        if w_bars.empty: continue
        
        ctx_at_open = w_bars['ctx_dir'].iloc[0]
        if np.isnan(ctx_at_open) or ctx_at_open == 0: continue
        direction = int(ctx_at_open)
        
        pre_window = day_bars[day_bars['hour_msk'] < ws]
        if direction == 1 and pre_window['high'].max() >= pdh: continue
        if direction == -1 and pre_window['low'].min() <= pdl: continue
        
        w_highs = w_bars['high'].to_numpy()
        w_lows = w_bars['low'].to_numpy()
        w_closes = w_bars['close'].to_numpy()
        w_opens = w_bars['open'].to_numpy()
        w_times = w_bars.index
        
        trade_taken = False
        
        if direction == 1:
            for ref_level in sweep_targets_lo:
                if trade_taken: break
                sweep_done = False
                sweep_ext = np.nan
                sweep_bar = -1
                origin_swing = np.nan
                for b in range(len(w_bars)):
                    if not sweep_done:
                        if w_lows[b] < ref_level - pen:
                            sweep_done = True
                            sweep_ext = w_lows[b]
                            sweep_bar = b
                            origin_swing = w_highs[max(0, b-15):b+1].max()
                    else:
                        if w_lows[b] < sweep_ext:
                            sweep_ext = w_lows[b]
                            origin_swing = w_highs[max(0, b-15):b+1].max()
                        if b <= sweep_bar + 25 and w_closes[b] > origin_swing and w_closes[b] > w_opens[b]:
                            entry_px = w_closes[b]
                            sl = sweep_ext - pen
                            risk = entry_px - sl
                            if risk >= min_risk:
                                cands = [v for v in [fra_hi, asia_hi, pdh] if (v - entry_px)/risk >= 1.8]
                                if cands:
                                    best_tp = min(cands)
                                    target_rr = (best_tp - entry_px)/risk
                                    if 1.8 <= target_rr <= 4.2:
                                        rem = day_bars.loc[w_times[b]:]
                                        rem_h = rem['high'].to_numpy()
                                        rem_l = rem['low'].to_numpy()
                                        win, loss = False, False
                                        for k in range(1, len(rem)):
                                            if rem_l[k] <= sl: loss = True; break
                                            elif rem_h[k] >= best_tp: win = True; break
                                        if win or loss:
                                            trades.append({
                                                'date': str(day)[:10],
                                                'symbol': sym,
                                                'dir': 'LONG',
                                                'result': 'TP' if win else 'SL',
                                                'rr': round(target_rr if win else -1.0, 2),
                                                'entry': entry_px, 'sl': sl, 'tp': best_tp
                                            })
                                            trade_taken = True
                                            break
        elif direction == -1:
            for ref_level in sweep_targets_hi:
                if trade_taken: break
                sweep_done = False
                sweep_ext = np.nan
                sweep_bar = -1
                origin_swing = np.nan
                for b in range(len(w_bars)):
                    if not sweep_done:
                        if w_highs[b] > ref_level + pen:
                            sweep_done = True
                            sweep_ext = w_highs[b]
                            sweep_bar = b
                            origin_swing = w_lows[max(0, b-15):b+1].min()
                    else:
                        if w_highs[b] > sweep_ext:
                            sweep_ext = w_highs[b]
                            origin_swing = w_lows[max(0, b-15):b+1].min()
                        if b <= sweep_bar + 25 and w_closes[b] < origin_swing and w_closes[b] < w_opens[b]:
                            entry_px = w_closes[b]
                            sl = sweep_ext + pen
                            risk = sl - entry_px
                            if risk >= min_risk:
                                cands = [v for v in [fra_lo, asia_lo, pdl] if (entry_px - v)/risk >= 1.8]
                                if cands:
                                    best_tp = max(cands)
                                    target_rr = (entry_px - best_tp)/risk
                                    if 1.8 <= target_rr <= 4.2:
                                        rem = day_bars.loc[w_times[b]:]
                                        rem_h = rem['high'].to_numpy()
                                        rem_l = rem['low'].to_numpy()
                                        win, loss = False, False
                                        for k in range(1, len(rem)):
                                            if rem_h[k] >= sl: loss = True; break
                                            elif rem_l[k] <= best_tp: win = True; break
                                        if win or loss:
                                            trades.append({
                                                'date': str(day)[:10],
                                                'symbol': sym,
                                                'dir': 'SHORT',
                                                'result': 'TP' if win else 'SL',
                                                'rr': round(target_rr if win else -1.0, 2),
                                                'entry': entry_px, 'sl': sl, 'tp': best_tp
                                            })
                                            trade_taken = True
                                            break
    df_res = pd.DataFrame(trades)
    print(f"Done {sym}: {len(df_res)} trades, winrate = {(df_res['result']=='TP').mean()*100 if not df_res.empty else 0:.1f}%, Total R = {df_res['rr'].sum() if not df_res.empty else 0:.2f}")
    return df_res

def main():
    t0 = time.time()
    dfs = []
    for s, spec in ASSET_SPECS.items():
        print(f"Processing {s}...")
        df_s = backtest_asset(s, spec, 2021, 2024)
        if not df_s.empty:
            dfs.append(df_s)
            
    all_df = pd.concat(dfs, ignore_index=True)
    all_df = all_df.sort_values('date').reset_index(drop=True)
    all_df['year'] = pd.to_datetime(all_df['date']).dt.year
    all_df['month'] = pd.to_datetime(all_df['date']).dt.to_period('M')
    all_df.to_csv('method4_complete_portfolio.csv', index=False)
    
    print(f"\nAll assets calculated in {time.time()-t0:.1f}s!")
    print("="*85)
    print("COMPLETE METHOD 4 SNIPER PORTFOLIO (2021 - 2024)")
    print("="*85)
    
    yearly = []
    for yr, grp in all_df.groupby('year'):
        tr = len(grp)
        wins = (grp['result'] == 'TP').sum()
        losses = tr - wins
        wr = wins / tr * 100
        tot_r = grp['rr'].sum()
        
        # Intra-year drawdown in R
        grp_r = grp['rr'].values
        eq = np.cumsum(grp_r)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd_r = dd.min() if len(dd) > 0 else 0
        
        # Monthly breakdown
        monthly_r = grp.groupby('month')['rr'].sum()
        worst_month = monthly_r.min()
        best_month = monthly_r.max()
        
        yearly.append({
            'Year': yr,
            'Trades': tr,
            'Wins': wins,
            'Losses': losses,
            'Winrate': f"{wr:.1f}%",
            'Total R (Profit %)': round(tot_r, 2),
            'Max Intra-Year DD (R)': round(max_dd_r, 2),
            'Worst Month (R)': round(worst_month, 2),
            'Best Month (R)': round(best_month, 2)
        })
        
    res_y = pd.DataFrame(yearly)
    print(res_y.to_string(index=False))
    
    print("\n" + "="*85)
    print("--- TOTAL SUMMARY (2021 - 2024) ---")
    tot_trades = len(all_df)
    tot_wins = (all_df['result'] == 'TP').sum()
    print(f"Total Trades: {tot_trades} (~{tot_trades/3.33:.1f} trades/year across 6 assets)")
    print(f"Portfolio Winrate: {tot_wins/tot_trades*100:.1f}%")
    print(f"Total Portfolio R: {all_df['rr'].sum():.2f} R (+{all_df['rr'].sum():.1f}%)")
    print(f"Avg Annual Return (1R = 1% = $100): +{all_df['rr'].sum()/3.33:.1f}% per year (+{all_df['rr'].sum()/3.33:.1f} R/year)")

if __name__ == '__main__':
    main()
