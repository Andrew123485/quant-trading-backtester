"""Fast Audit Method 4 Sniper on EURUSD across all 17 years (2008-2024)
Optimized with groupby('msk_date') for super-fast execution.
"""
import sys, os, time
import numpy as np
import pandas as pd
import data_loader
from strategies import context_h1

def audit_eurusd_longterm():
    t0 = time.time()
    print("Loading data...")
    m1 = pd.read_pickle('data_m1.pkl')
    lo = pd.Timestamp("2008-01-01", tz='UTC')
    hi = pd.Timestamp("2024-05-01 02:59:00", tz='UTC')
    
    m1_sub = m1[(m1.index >= lo - pd.Timedelta(days=20)) & (m1.index <= hi)].copy()
    ohlc = m1_sub[['open', 'high', 'low', 'close']]
    h1 = data_loader._resample(ohlc, '1h')
    d1 = data_loader._resample(ohlc, '1D')
    
    m1_eval = m1[(m1.index >= lo) & (m1.index <= hi)].copy()
    m1_eval = data_loader.add_msk_columns(m1_eval)
    m1_eval['msk_date'] = (m1_eval.index + pd.Timedelta(hours=3)).normalize()
    
    pip = 0.0001
    min_risk = 0.00045
    
    print("Detecting H1 context...")
    ctx = context_h1.detect(h1, penetration=pip)
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
    
    print("Iterating through days via groupby...")
    trades = []
    
    # Fast iteration by groupby
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
        
        # SNIPER FILTER 1: Late Frankfurt Trap
        fra_last_10 = fra[fra['minute_msk'] >= 50]
        if not fra_last_10.empty:
            if fra_last_10['high'].max() >= fra_hi or fra_last_10['low'].min() <= fra_lo:
                continue # Late trap
                
        ws, we = lon_start, lon_end
        ref_hi, ref_lo = fra_hi, fra_lo
        
        w_bars = day_bars[(day_bars['hour_msk'] >= ws) & (day_bars['hour_msk'] < we)]
        if w_bars.empty: continue
        
        ctx_at_open = w_bars['ctx_dir'].iloc[0]
        if np.isnan(ctx_at_open) or ctx_at_open == 0: continue
        direction = int(ctx_at_open)
        
        # SNIPER FILTER 2: PDH / PDL already swept check
        pre_window = day_bars[day_bars['hour_msk'] < ws]
        if direction == 1 and pre_window['high'].max() >= pdh:
            continue
        if direction == -1 and pre_window['low'].min() <= pdl:
            continue
            
        w_highs = w_bars['high'].to_numpy()
        w_lows = w_bars['low'].to_numpy()
        w_closes = w_bars['close'].to_numpy()
        w_opens = w_bars['open'].to_numpy()
        w_times = w_bars.index
        
        sweep_done = False
        sweep_ext = np.nan
        sweep_bar = -1
        origin_swing = np.nan
        pen = pip
        
        if direction == 1:
            for b in range(len(w_bars)):
                if not sweep_done:
                    if w_lows[b] < ref_lo - pen:
                        sweep_done = True
                        sweep_ext = w_lows[b]
                        sweep_bar = b
                        origin_swing = w_highs[max(0, b-15):b+1].max()
                else:
                    if w_lows[b] < sweep_ext:
                        sweep_ext = w_lows[b]
                        origin_swing = w_highs[max(0, b-15):b+1].max()
                    # SNIPER FILTER 3: Displacement body close
                    if b <= sweep_bar + 25 and w_closes[b] > origin_swing and w_closes[b] > w_opens[b]:
                        entry_px = w_closes[b]
                        sl = sweep_ext - pen
                        risk = entry_px - sl
                        if risk >= min_risk:
                            cands = [v for v in [ref_hi, asia_hi, pdh] if (v - entry_px) / risk >= 1.8]
                            if cands:
                                best_tp = min(cands)
                                target_rr = (best_tp - entry_px) / risk
                                if 1.8 <= target_rr <= 4.2:
                                    tp = best_tp
                                    rem = day_bars.loc[w_times[b]:]
                                    rem_h = rem['high'].to_numpy()
                                    rem_l = rem['low'].to_numpy()
                                    win = False; loss = False
                                    for k in range(1, len(rem)):
                                        if rem_l[k] <= sl:
                                            loss = True; break
                                        elif rem_h[k] >= tp:
                                            win = True; break
                                    if win or loss:
                                        trades.append({
                                            'date': str(day)[:10],
                                            'symbol': 'EURUSD',
                                            'dir': 'LONG',
                                            'result': 'TP' if win else 'SL',
                                            'rr': round((tp - entry_px) / risk if win else -1.0, 2),
                                            'entry': entry_px, 'sl': sl, 'tp': tp
                                        })
                                        break
        elif direction == -1:
            for b in range(len(w_bars)):
                if not sweep_done:
                    if w_highs[b] > ref_hi + pen:
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
                            cands = [v for v in [ref_lo, asia_lo, pdl] if (entry_px - v) / risk >= 1.8]
                            if cands:
                                best_tp = max(cands)
                                target_rr = (entry_px - best_tp) / risk
                                if 1.8 <= target_rr <= 4.2:
                                    tp = best_tp
                                    rem = day_bars.loc[w_times[b]:]
                                    rem_h = rem['high'].to_numpy()
                                    rem_l = rem['low'].to_numpy()
                                    win = False; loss = False
                                    for k in range(1, len(rem)):
                                        if rem_h[k] >= sl:
                                            loss = True; break
                                        elif rem_l[k] <= tp:
                                            win = True; break
                                    if win or loss:
                                        trades.append({
                                            'date': str(day)[:10],
                                            'symbol': 'EURUSD',
                                            'dir': 'SHORT',
                                            'result': 'TP' if win else 'SL',
                                            'rr': round((entry_px - tp) / risk if win else -1.0, 2),
                                            'entry': entry_px, 'sl': sl, 'tp': tp
                                        })
                                        break
                                        
    df_t = pd.DataFrame(trades)
    df_t.to_csv('eurusd_sniper_17y.csv', index=False)
    df_t['year'] = pd.to_datetime(df_t['date']).dt.year
    df_t['month'] = pd.to_datetime(df_t['date']).dt.to_period('M')
    
    print(f"Calculations done in {time.time() - t0:.1f}s!")
    print("\n" + "="*85)
    print("EURUSD METHOD 4 (SNIPER QUALITY) - 17 YEARS (2008 - 2024)")
    print("="*85)
    
    yearly = []
    for yr, grp in df_t.groupby('year'):
        tr = len(grp)
        wins = (grp['result'] == 'TP').sum()
        wr = wins / tr * 100
        tot_r = grp['rr'].sum()
        
        # Drawdown in R within the year
        grp_r = grp['rr'].values
        eq = np.cumsum(grp_r)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        max_dd_r = dd.min() if len(dd) > 0 else 0
        
        # Max monthly drawdown
        monthly_r = grp.groupby('month')['rr'].sum()
        worst_month_r = monthly_r.min() if len(monthly_r) > 0 else 0
        best_month_r = monthly_r.max() if len(monthly_r) > 0 else 0
        
        yearly.append({
            'Год': yr,
            'Сделок': tr,
            'Тейков': wins,
            'Стопов': tr - wins,
            'Винрейт': f"{wr:.1f}%",
            'Прибыль R (%)': round(tot_r, 2),
            'Макс. DD (R)': round(max_dd_r, 2),
            'Худший мес. (R)': round(worst_month_r, 2),
            'Лучший мес. (R)': round(best_month_r, 2)
        })
        
    res_y = pd.DataFrame(yearly)
    print(res_y.to_string(index=False))
    
    print("\n" + "="*85)
    print("--- ИТОГИ EURUSD ЗА ВСЕ 17 ЛЕТ (2008 - 2024) ---")
    tot_trades = len(df_t)
    tot_wins = (df_t['result'] == 'TP').sum()
    print(f"Всего сделок: {tot_trades} (в среднем {tot_trades/len(yearly):.1f} в год)")
    print(f"Общий винрейт: {tot_wins/tot_trades*100:.1f}%")
    print(f"Суммарный R: {df_t['rr'].sum():.2f} R (+{df_t['rr'].sum():.1f}%)")
    print(f"Средняя прибыль в год: {df_t['rr'].sum()/len(yearly):.2f} R/год")
    
    # Check consecutive losses
    streak = 0
    max_streak = 0
    for res in df_t['result']:
        if res == 'SL':
            streak += 1
            if streak > max_streak: max_streak = streak
        else:
            streak = 0
    print(f"Максимальная серия стопов подряд за 17 лет: {max_streak} SL")

if __name__ == '__main__':
    audit_eurusd_longterm()
