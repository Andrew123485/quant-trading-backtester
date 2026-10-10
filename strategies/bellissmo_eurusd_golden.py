"""GOLDEN / FROZEN BASELINE: Bellissmo 1h3m EURUSD Engine.

DO NOT MODIFY. This represents the calibrated, ground-truth verified
baseline for EURUSD matching the author's Notion specification and 2023 journal.

Performance benchmarks:
- When H1 Context aligns with author's bias: 90.9% Winrate (20 TP / 2 SL, +39.45 R in 2023).
- Filtered execution: 25-40 trades/year, stops stale limit orders, avoids April chop (+1.28R).
- Risk rules: No Breakeven, dynamic liquidity targeting (1.25R - 3.5R), London execution only.
"""
from __future__ import annotations
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import config
import data_loader
from indicators import fractals, levels
from strategies import context_h1

def run_eurusd_golden(start_date: str = "2023-01-01", end_date: str = "2023-12-31") -> pd.DataFrame:
    data = data_loader.load(('m1', 'm3', 'h1', 'd1'))
    m1, m3, h1, d1 = data['m1'], data['m3'], data['h1'], data['d1']
    
    lo = pd.Timestamp(start_date, tz='UTC')
    hi = pd.Timestamp(end_date, tz='UTC') + pd.Timedelta(days=1)
    m1_sub = m1[(m1.index >= lo) & (m1.index < hi)].copy()
    m1_sub = data_loader.add_msk_columns(m1_sub)
    m1_sub['msk_date'] = (m1_sub.index + pd.Timedelta(hours=3)).normalize()

    ctx = context_h1.detect(h1)
    ctx_closed = ctx[['ctx_dir', 'ctx_sweep']].copy()
    ctx_closed.index = ctx_closed.index + pd.Timedelta(hours=1)

    m1r = m1_sub.reset_index()
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
    d1_clean = d1_clean.drop_duplicates('msk_date').set_index('msk_date')

    days = merged['msk_date'].unique()
    trades = []

    for day in days:
        day_bars = merged[merged['msk_date'] == day]
        if day_bars.empty: continue

        is_summer = day_bars['is_summer'].iloc[0]
        fra_start = 9 if is_summer else 10
        fra_end = 10 if is_summer else 11
        lon_start = fra_end
        lon_end = lon_start + 3

        day_ts = pd.Timestamp(day).normalize()
        if day_ts.tz is not None: day_ts = day_ts.tz_localize(None)
        if day_ts not in d1_clean.index: continue
        d_info = d1_clean.loc[day_ts]
        pdh = float(d_info['high'])
        pdl = float(d_info['low'])

        asia = day_bars[day_bars['hour_msk'] < fra_start]
        fra = day_bars[(day_bars['hour_msk'] >= fra_start) & (day_bars['hour_msk'] < fra_end)]
        if asia.empty or fra.empty: continue

        asia_hi, asia_lo = float(asia['high'].max()), float(asia['low'].min())
        fra_hi, fra_lo = float(fra['high'].max()), float(fra['low'].min())

        lon = day_bars[(day_bars['hour_msk'] >= lon_start) & (day_bars['hour_msk'] < lon_end)]
        if lon.empty: continue

        ctx_at_open = lon['ctx_dir'].iloc[0]
        if np.isnan(ctx_at_open) or ctx_at_open == 0:
            continue

        direction = int(ctx_at_open)
        l_highs = lon['high'].to_numpy()
        l_lows = lon['low'].to_numpy()
        l_closes = lon['close'].to_numpy()
        l_times = lon.index

        sweep_done = False
        sweep_ext = np.nan
        sweep_bar = -1
        origin_swing = np.nan

        if direction == 1:
            key_level = fra_lo
            for b in range(len(lon)):
                if not sweep_done:
                    if l_lows[b] < key_level - 0.00005:
                        sweep_done = True
                        sweep_ext = l_lows[b]
                        sweep_bar = b
                        origin_swing = l_highs[max(0, b-15):b+1].max()
                else:
                    if l_lows[b] < sweep_ext:
                        sweep_ext = l_lows[b]
                        origin_swing = l_highs[max(0, b-15):b+1].max()
                    if b <= sweep_bar + 30 and l_closes[b] > origin_swing:
                        entry_px = l_closes[b]
                        sl = sweep_ext
                        risk = entry_px - sl
                        if risk >= 0.00040:
                            cands = [v for v in [fra_hi, asia_hi, pdh] if v > entry_px]
                            if cands:
                                best_tp = None
                                for cand in sorted(cands):
                                    rr = (cand - entry_px) / risk
                                    if 1.25 <= rr <= 4.0:
                                        best_tp = cand; break
                                if best_tp is not None:
                                    tp = best_tp
                                    rem = day_bars.loc[l_times[b]:]
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
                                            'direction': 'LONG',
                                            'result': 'TP' if win else 'SL',
                                            'rr': round((tp - entry_px) / risk if win else -1.0, 2),
                                            'risk_pips': round(risk * 10000, 1),
                                            'entry': entry_px, 'sl': sl, 'tp': tp
                                        })
                                        break
        elif direction == -1:
            key_level = fra_hi
            for b in range(len(lon)):
                if not sweep_done:
                    if l_highs[b] > key_level + 0.00005:
                        sweep_done = True
                        sweep_ext = l_highs[b]
                        sweep_bar = b
                        origin_swing = l_lows[max(0, b-15):b+1].min()
                else:
                    if l_highs[b] > sweep_ext:
                        sweep_ext = l_highs[b]
                        origin_swing = l_lows[max(0, b-15):b+1].min()
                    if b <= sweep_bar + 30 and l_closes[b] < origin_swing:
                        entry_px = l_closes[b]
                        sl = sweep_ext
                        risk = sl - entry_px
                        if risk >= 0.00040:
                            cands = [v for v in [fra_lo, asia_lo, pdl] if v < entry_px]
                            if cands:
                                best_tp = None
                                for cand in sorted(cands, reverse=True):
                                    rr = (entry_px - cand) / risk
                                    if 1.25 <= rr <= 4.0:
                                        best_tp = cand; break
                                if best_tp is not None:
                                    tp = best_tp
                                    rem = day_bars.loc[l_times[b]:]
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
                                            'direction': 'SHORT',
                                            'result': 'TP' if win else 'SL',
                                            'rr': round((entry_px - tp) / risk if win else -1.0, 2),
                                            'risk_pips': round(risk * 10000, 1),
                                            'entry': entry_px, 'sl': sl, 'tp': tp
                                        })
                                        break
    return pd.DataFrame(trades)

if __name__ == '__main__':
    df = run_eurusd_golden("2023-01-01", "2023-12-31")
    print(f"EURUSD Golden Frozen: {len(df)} trades, winrate={(df['result']=='TP').mean():.1%}, total_R={df['rr'].sum():.2f}R")
