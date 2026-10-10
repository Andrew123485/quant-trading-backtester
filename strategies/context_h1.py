"""H1 "1-2-3" context detector (rolling / self-refreshing).

Model (confirmed with the trader):
  (1) Sweep  - price takes a *recent* confirmed fractal (grabs liquidity).
  (2) BoS    - a clear break of the opposite structure fractal (body close).
  (3) Correction - first opposite fractal after the BoS (the pullback swing);
      the H1 context stays valid only while the sweep extreme (point 1) is NOT
      crossed by a close, and only for a limited number of bars (freshness).

Counting matches the trader: every Fractals(2) fractal is eligible, and the
ACTIVE context is always the latest completed 1-2-3.

Outputs (known at each bar's close, no lookahead):
  ctx_dir, ctx_sweep, ctx_sweep_level, ctx_bos_level, ctx_since
  ctx_p1_bar/level (swept fractal), ctx_p2_bar/level (BoS fractal),
  ctx_p3_bar/level (correction fractal)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import config
from indicators import fractals


def detect(h1: pd.DataFrame, period: int | None = None,
           penetration: float | None = None,
           max_age: int | None = None,
           sweep_timeout: int | None = None) -> pd.DataFrame:
    period = config.FRACTAL_PERIOD if period is None else period
    pen = config.MIN_SWEEP_PENETRATION if penetration is None else penetration
    max_age = config.CONTEXT_MAX_AGE_H1_BARS if max_age is None else max_age
    sweep_to = config.SWEEP_TIMEOUT_H1_BARS if sweep_timeout is None else sweep_timeout

    df = fractals.compute(h1, period=period)
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    fh_center = df["fh_center"].to_numpy()
    fl_center = df["fl_center"].to_numpy()
    n = len(df)

    ctx_dir = np.zeros(n, dtype=np.int8)
    ctx_sweep = np.full(n, np.nan)
    ctx_sweep_level = np.full(n, np.nan)
    ctx_bos_level = np.full(n, np.nan)
    ctx_since = np.full(n, -1, dtype=np.int64)
    p1_bar = np.full(n, -1, dtype=np.int64); p1_lvl = np.full(n, np.nan)
    p2_bar = np.full(n, -1, dtype=np.int64); p2_lvl = np.full(n, np.nan)
    p3_bar = np.full(n, -1, dtype=np.int64); p3_lvl = np.full(n, np.nan)

    last_fh = np.nan; last_fl = np.nan
    last_fh_bar = -1; last_fl_bar = -1

    s_on = False; s_ext = np.nan; s_level = np.nan; s_origin_low = np.nan
    s_bar = -1; s_p1_bar = -1; s_p2_bar = -1
    l_on = False; l_ext = np.nan; l_level = np.nan; l_origin_high = np.nan
    l_bar = -1; l_p1_bar = -1; l_p2_bar = -1

    cur_dir = 0; cur_sweep = np.nan; cur_level = np.nan; cur_bos = np.nan; cur_since = -1
    cur_p1_bar = -1; cur_p1_lvl = np.nan; cur_p2_bar = -1; cur_p2_lvl = np.nan
    cur_p3_bar = -1; cur_p3_lvl = np.nan

    for i in range(n):
        c = i - period
        if c >= 0:
            if fh_center[c]:
                last_fh = high[c]; last_fh_bar = c
                if cur_dir == 1 and cur_p3_bar == -1 and c > cur_since:
                    pass  # long correction is a LOW, not a high
                if cur_dir == -1 and cur_p3_bar == -1 and c > cur_since:
                    cur_p3_bar = c; cur_p3_lvl = high[c]
            if fl_center[c]:
                last_fl = low[c]; last_fl_bar = c
                if cur_dir == 1 and cur_p3_bar == -1 and c > cur_since:
                    cur_p3_bar = c; cur_p3_lvl = low[c]

        hi, lo, cl = high[i], low[i], close[i]

        # ---- expire / invalidate current context ----
        if cur_dir != 0:
            if i - cur_since > max_age:
                cur_dir = 0
            elif cur_dir == -1 and cl > cur_sweep:
                cur_dir = 0
            elif cur_dir == 1 and cl < cur_sweep:
                cur_dir = 0

        # ---- SHORT sweep -> BoS ----
        if not s_on:
            if not np.isnan(last_fh) and hi > last_fh + pen and not np.isnan(last_fl):
                s_on = True; s_ext = hi; s_level = last_fh; s_origin_low = last_fl
                s_bar = i; s_p1_bar = last_fh_bar; s_p2_bar = last_fl_bar
        else:
            if hi > s_ext:
                s_ext = hi
            if cl < s_origin_low:
                cur_dir = -1; cur_sweep = s_ext; cur_level = s_level
                cur_bos = s_origin_low; cur_since = i
                cur_p1_bar = s_p1_bar; cur_p1_lvl = s_level
                cur_p2_bar = s_p2_bar; cur_p2_lvl = s_origin_low
                cur_p3_bar = -1; cur_p3_lvl = np.nan
                s_on = False
            elif i - s_bar > sweep_to:
                s_on = False

        # ---- LONG sweep -> BoS ----
        if not l_on:
            if not np.isnan(last_fl) and lo < last_fl - pen and not np.isnan(last_fh):
                l_on = True; l_ext = lo; l_level = last_fl; l_origin_high = last_fh
                l_bar = i; l_p1_bar = last_fl_bar; l_p2_bar = last_fh_bar
        else:
            if lo < l_ext:
                l_ext = lo
            if cl > l_origin_high:
                cur_dir = 1; cur_sweep = l_ext; cur_level = l_level
                cur_bos = l_origin_high; cur_since = i
                cur_p1_bar = l_p1_bar; cur_p1_lvl = l_level
                cur_p2_bar = l_p2_bar; cur_p2_lvl = l_origin_high
                cur_p3_bar = -1; cur_p3_lvl = np.nan
                l_on = False
            elif i - l_bar > sweep_to:
                l_on = False

        active = cur_dir != 0
        ctx_dir[i] = cur_dir
        ctx_sweep[i] = cur_sweep if active else np.nan
        ctx_sweep_level[i] = cur_level if active else np.nan
        ctx_bos_level[i] = cur_bos if active else np.nan
        ctx_since[i] = cur_since if active else -1
        if active:
            p1_bar[i] = cur_p1_bar; p1_lvl[i] = cur_p1_lvl
            p2_bar[i] = cur_p2_bar; p2_lvl[i] = cur_p2_lvl
            p3_bar[i] = cur_p3_bar; p3_lvl[i] = cur_p3_lvl

    out = df
    out["ctx_dir"] = ctx_dir
    out["ctx_sweep"] = ctx_sweep
    out["ctx_sweep_level"] = ctx_sweep_level
    out["ctx_bos_level"] = ctx_bos_level
    out["ctx_since"] = ctx_since
    out["ctx_p1_bar"] = p1_bar; out["ctx_p1_level"] = p1_lvl
    out["ctx_p2_bar"] = p2_bar; out["ctx_p2_level"] = p2_lvl
    out["ctx_p3_bar"] = p3_bar; out["ctx_p3_level"] = p3_lvl
    return out
