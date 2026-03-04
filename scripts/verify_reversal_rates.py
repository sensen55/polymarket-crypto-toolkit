#!/usr/bin/env python3
"""
Verify streak reversal rates using Binance BTCUSDT candle data.

Compares the hardcoded REVERSAL_RATES in src/strategies/streak.py against
actual historical data spanning multiple years.

Usage:
    uv run python scripts/verify_reversal_rates.py
"""

import io
import sys
import time
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd
import requests

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Hardcoded values from src/strategies/streak.py
REPO_REVERSAL_RATES = {
    2: 0.540,
    3: 0.579,
    4: 0.667,
    5: 0.824,
}

DATA_DIR = Path("data")


def fetch_5m_from_binance_vision() -> pd.DataFrame | None:
    """Fetch BTCUSDT 5m data from Binance public data (data.binance.vision).

    This endpoint does not require API auth and is not blocked by proxy.
    Downloads monthly kline ZIP files.
    """
    base_url = "https://data.binance.vision/data/spot/monthly/klines/BTCUSDT/5m"
    columns = [
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_asset_volume", "number_of_trades",
        "taker_buy_base_asset_volume", "taker_buy_quote_asset_volume", "ignore",
    ]

    all_frames = []
    # Fetch from 2020-01 to 2026-02
    for year in range(2020, 2027):
        for month in range(1, 13):
            if year == 2026 and month >= 3:
                break
            if year == 2020 and month < 1:
                continue

            label = f"{year}-{month:02d}"
            url = f"{base_url}/BTCUSDT-5m-{label}.zip"

            try:
                resp = requests.get(url, timeout=30)
                if resp.status_code == 404:
                    continue
                resp.raise_for_status()

                with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
                    csv_name = zf.namelist()[0]
                    with zf.open(csv_name) as f:
                        df = pd.read_csv(f, header=None, names=columns)
                        all_frames.append(df)
                        print(f"    {label}: {len(df):,} candles")

            except Exception as e:
                print(f"    {label}: skipped ({e})")
                continue

    if not all_frames:
        return None

    df = pd.concat(all_frames, ignore_index=True)
    # Ensure open_time is numeric before datetime conversion to filter bad values
    df["open_time"] = pd.to_numeric(df["open_time"], errors="coerce")
    # Filter out timestamps outside reasonable range (2015-01-01 to 2030-01-01 in ms)
    min_ts = int(datetime(2015, 1, 1, tzinfo=UTC).timestamp() * 1000)
    max_ts = int(datetime(2030, 1, 1, tzinfo=UTC).timestamp() * 1000)
    df = df[(df["open_time"] >= min_ts) & (df["open_time"] <= max_ts)]
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)
    return df


def load_local_4h() -> pd.DataFrame:
    """Load cached BTC 4h data as fallback."""
    path = DATA_DIR / "btc_4h.csv"
    print(f"[+] Loading local BTC 4h data from {path}")
    df = pd.read_csv(path)
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    print(f"    {len(df):,} candles loaded")
    return df


def fetch_data() -> tuple[pd.DataFrame, str]:
    """Fetch data, trying 5m first, falling back to 4h.

    Returns (dataframe, timeframe_label).
    """
    cache_5m = DATA_DIR / "btcusdt_5m.parquet"

    # Try cached 5m data
    if cache_5m.exists():
        print(f"[+] Loading cached 5m data from {cache_5m}")
        df = pd.read_parquet(cache_5m)
        # Filter out any rows with out-of-range timestamps
        times = pd.to_datetime(df["open_time"], utc=True)
        min_date = pd.Timestamp("2015-01-01", tz="UTC")
        max_date = pd.Timestamp("2030-01-01", tz="UTC")
        bad = (times < min_date) | (times > max_date)
        if bad.any():
            print(f"    Filtered {bad.sum():,} rows with out-of-range timestamps")
            df = df[~bad].reset_index(drop=True)
        print(f"    {len(df):,} candles loaded")
        return df, "5m"

    # Try fetching 5m from Binance public data
    print("[+] Fetching BTCUSDT 5m from data.binance.vision (public data)...")
    df = fetch_5m_from_binance_vision()
    if df is not None and len(df) > 10000:
        # Save cache
        cache_5m.parent.mkdir(exist_ok=True)
        df.to_parquet(cache_5m, index=False)
        print(f"    Saved {len(df):,} candles to {cache_5m}")
        return df, "5m"

    # Fallback to local 4h data
    print("[!] Could not fetch 5m data, falling back to local 4h data")
    return load_local_4h(), "4h"


def analyze_streaks(df: pd.DataFrame) -> dict:
    """Analyze reversal rates for each streak length."""
    close = df["close"].astype(float)
    open_ = df["open"].astype(float)
    direction = (close > open_).astype(int).replace(0, -1)  # 1=up, -1=down

    # Handle doji (close == open) as continuation of previous direction
    doji_mask = close == open_
    doji_count = doji_mask.sum()
    if doji_count > 0:
        direction[doji_mask] = pd.NA
        direction = direction.ffill().fillna(1).astype(int)
        print(f"    Doji candles (close==open): {doji_count:,} → treated as continuation")

    directions = direction.values
    n = len(directions)

    # Count streaks and what happens after
    results = {}  # streak_length -> {"reversals": int, "continuations": int}

    i = 0
    while i < n:
        streak_dir = directions[i]
        streak_len = 1
        while i + streak_len < n and directions[i + streak_len] == streak_dir:
            streak_len += 1

        # For each sub-streak length >= 2, record what happened after
        for length in range(2, streak_len + 1):
            after_idx = i + length
            if after_idx < n:
                next_dir = directions[after_idx]
                if length not in results:
                    results[length] = {"reversals": 0, "continuations": 0}
                if next_dir != streak_dir:
                    results[length]["reversals"] += 1
                else:
                    results[length]["continuations"] += 1

        i += streak_len

    return results


def print_results(results: dict, total_candles: int, date_range: str, timeframe: str):
    """Print formatted comparison table."""
    tf_label = "5-Minute" if timeframe == "5m" else "4-Hour"
    print("\n" + "=" * 85)
    print(f"STREAK REVERSAL RATE VERIFICATION — BTCUSDT {tf_label} Candles")
    print(f"Data: {total_candles:,} candles | Period: {date_range}")
    print("=" * 85)

    print(
        f"\n{'Streak':>8} | {'Samples':>9} | {'Reversals':>10} | {'Actual Rate':>12} | "
        f"{'Repo Rate':>10} | {'Difference':>11} | {'Verdict':>10}"
    )
    print("-" * 85)

    for streak_len in sorted(results.keys()):
        if streak_len < 2 or streak_len > 10:
            continue

        data = results[streak_len]
        total = data["reversals"] + data["continuations"]
        if total == 0:
            continue

        actual_rate = data["reversals"] / total
        repo_rate = REPO_REVERSAL_RATES.get(min(streak_len, 5))

        if repo_rate:
            diff = actual_rate - repo_rate
            diff_str = f"{diff:+.1%}"
            if abs(diff) < 0.02:
                verdict = "MATCH"
            elif abs(diff) < 0.05:
                verdict = "CLOSE"
            else:
                verdict = "MISMATCH"
        else:
            diff_str = "N/A"
            verdict = "—"

        repo_rate_str = f"{repo_rate:.1%}" if repo_rate else "—"

        print(
            f"{streak_len:>8} | {total:>9,} | {data['reversals']:>10,} | {actual_rate:>11.1%} | "
            f"{repo_rate_str:>10} | {diff_str:>11} | {verdict:>10}"
        )

    print("\n" + "-" * 85)
    print("Repo Rate: from src/strategies/streak.py (570-market, 2-day Polymarket backtest)")
    print("Actual Rate: from Binance BTCUSDT historical candles (multi-year data)")
    print()
    print("BASELINE: A fair coin would show ~50% reversal at every streak length.")
    print("If actual rates are near 50%, streaks have NO predictive power.")
    print()


def main():
    df, timeframe = fetch_data()

    # Date range — filter to valid range to avoid out-of-range Timestamp errors
    if "open_time" in df.columns:
        times = pd.to_datetime(df["open_time"], utc=True)
        min_date = pd.Timestamp("2015-01-01", tz="UTC")
        max_date = pd.Timestamp("2030-01-01", tz="UTC")
        valid = times[(times >= min_date) & (times <= max_date)]
        if len(valid) > 0:
            date_range = f"{valid.min().date()} to {valid.max().date()}"
            # Also filter the dataframe itself
            mask = (times >= min_date) & (times <= max_date)
            df = df[mask].reset_index(drop=True)
        else:
            date_range = "unknown"
    else:
        date_range = "unknown"

    print(f"\n[+] Analyzing streak reversal rates...")
    results = analyze_streaks(df)
    print_results(results, len(df), date_range, timeframe)


if __name__ == "__main__":
    main()
