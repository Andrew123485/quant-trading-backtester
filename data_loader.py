"""Load raw M1 CSV and build M1/M3/H1/D1/W1 frames with MSK + session info.

Timestamps in the raw CSV are UNIX milliseconds in UTC. We keep a tz-aware UTC
index and add MSK (fixed UTC+3) helper columns. Summer/winter is derived from
European DST (last Sunday of March .. last Sunday of October, 01:00 UTC).
"""
from __future__ import annotations

import os
import time
import numpy as np
import pandas as pd

import config


def _last_sunday(year: int, month: int) -> pd.Timestamp:
    d = pd.Timestamp(year=year, month=month, day=1, tz="UTC") + pd.offsets.MonthEnd(0)
    while d.dayofweek != 6:
        d -= pd.Timedelta(days=1)
    return d.normalize() + pd.Timedelta(hours=1)  # 01:00 UTC


def european_summer_mask(utc_index: pd.DatetimeIndex) -> np.ndarray:
    """Vectorized EU DST flag (True = summer/CEST)."""
    out = np.zeros(len(utc_index), dtype=bool)
    years = np.unique(utc_index.year)
    for y in years:
        start = _last_sunday(int(y), 3)
        end = _last_sunday(int(y), 10)
        m = (utc_index >= start) & (utc_index < end)
        out |= np.asarray(m)
    return out


def add_msk_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add is_summer, time_msk, hour_msk, minute_msk, in_window, ny_open_hour."""
    utc_index = df.index
    is_summer = european_summer_mask(utc_index)
    df["is_summer"] = is_summer

    time_msk = utc_index + pd.Timedelta(hours=config.MSK_OFFSET_HOURS)
    df["hour_msk"] = time_msk.hour
    df["minute_msk"] = time_msk.minute
    df["dow"] = time_msk.dayofweek  # 0=Mon .. 6=Sun

    start = np.where(is_summer, config.WINDOW_SUMMER[0], config.WINDOW_WINTER[0])
    end = np.where(is_summer, config.WINDOW_SUMMER[1], config.WINDOW_WINTER[1])
    hour = df["hour_msk"].to_numpy()
    df["in_window"] = (hour >= start) & (hour < end)
    df["ny_open_hour"] = np.where(is_summer, config.NY_OPEN_SUMMER, config.NY_OPEN_WINTER)
    return df


def _resample(df_m1: pd.DataFrame, rule: str, origin=None) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    kw = dict(label="left", closed="left")
    if origin is not None:
        kw["origin"] = origin
    return df_m1.resample(rule, **kw).agg(agg).dropna()


def msk_midnight_origin(idx: pd.DatetimeIndex) -> pd.Timestamp:
    """2007-01-01 00:00 MSK = 2006-12-31 21:00 UTC. H4/M15/M5 must use this, not UTC midnight."""
    o = pd.Timestamp("2006-12-31 21:00")
    tz = getattr(idx, "tz", None)
    return o.tz_localize(tz) if tz is not None else o


def resample_msk(df_m1: pd.DataFrame, rule: str) -> pd.DataFrame:
    return _resample(df_m1, rule, origin=msk_midnight_origin(df_m1.index))


def build(raw_csv: str | None = None, save: bool = True) -> dict[str, pd.DataFrame]:
    raw_csv = raw_csv or config.RAW_M1_CSV
    t0 = time.time()
    print(f"Reading {raw_csv} ...")
    df = pd.read_csv(raw_csv)
    df["time"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("time").drop(columns=["timestamp"], errors="ignore")
    df = df[["open", "high", "low", "close"]].sort_index()
    print(f"  M1 rows: {len(df):,} ({time.time()-t0:.1f}s)")

    frames = {"m1": df}
    frames["m3"] = _resample(df, "3min")
    frames["h1"] = _resample(df, "1h")
    frames["d1"] = _resample(df, "1D")
    frames["w1"] = _resample(df, "1W")
    print(f"  M3: {len(frames['m3']):,}  H1: {len(frames['h1']):,}  "
          f"D1: {len(frames['d1']):,}  W1: {len(frames['w1']):,}")

    for k in ("m1", "m3", "h1"):
        frames[k] = add_msk_columns(frames[k])

    if save:
        for k, f in frames.items():
            path = config.PKL[k]
            f.to_pickle(path)
            print(f"  saved {path} ({len(f):,} rows)")
    print(f"Done in {time.time()-t0:.1f}s")
    return frames


def load(which=("m3", "h1", "d1", "w1")) -> dict[str, pd.DataFrame]:
    out = {}
    for k in which:
        out[k] = pd.read_pickle(config.PKL[k])
    return out


if __name__ == "__main__":
    need_build = not all(os.path.exists(config.PKL[k]) for k in ("m3", "h1", "d1", "w1"))
    if need_build:
        build()
    else:
        print("Pickles already present. Delete them to rebuild.")
