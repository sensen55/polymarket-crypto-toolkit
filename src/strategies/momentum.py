"""
Live BTC momentum signal — fetches recent Binance candles and evaluates
the BTCMomentumStrategy for use with bot.py.

Returns a Signal compatible with src/strategies/streak.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

import pandas as pd
import requests

BINANCE_KLINES = "https://api.binance.com/api/v3/klines"


@dataclass
class Signal:
    """Trading signal (same shape as streak.Signal)."""

    should_bet: bool
    direction: str  # "up" or "down"
    streak_length: int  # unused for momentum, set to 0
    streak_direction: str  # unused for momentum
    confidence: float
    reason: str


def fetch_recent_candles(
    symbol: str = "BTCUSDT",
    interval: str = "5m",
    count: int = 50,
) -> pd.DataFrame:
    """Fetch the most recent *count* candles from Binance."""
    resp = requests.get(
        BINANCE_KLINES,
        params={"symbol": symbol, "interval": interval, "limit": count},
        timeout=10,
    )
    resp.raise_for_status()
    rows = resp.json()

    df = pd.DataFrame(
        rows,
        columns=[
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ],
    )
    for col in ("open", "high", "low", "close", "volume"):
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df = df.set_index("open_time").sort_index()
    return df


def evaluate(
    _outcomes: list[str] | None = None,
    *,
    ema_fast: int = 8,
    ema_slow: int = 21,
    rsi_period: int = 14,
    momentum_window: int = 3,
    **_kwargs,
) -> Signal:
    """
    Evaluate BTC momentum signal using live Binance data.

    Parameters match src/strategies/streak.evaluate() signature so
    bot.py can call either one.  The *_outcomes* argument is accepted
    but ignored (momentum uses price data, not market outcomes).
    """
    try:
        candles = fetch_recent_candles(count=50)
    except Exception as e:
        return Signal(
            should_bet=False,
            direction="",
            streak_length=0,
            streak_direction="",
            confidence=0,
            reason=f"Failed to fetch candles: {e}",
        )

    if len(candles) < ema_slow + 5:
        return Signal(
            should_bet=False,
            direction="",
            streak_length=0,
            streak_direction="",
            confidence=0,
            reason=f"Not enough candles ({len(candles)})",
        )

    close = candles["close"]

    # EMA trend
    ema_f = close.ewm(span=ema_fast, adjust=False).mean()
    ema_s = close.ewm(span=ema_slow, adjust=False).mean()
    trend_up = bool(ema_f.iloc[-1] > ema_s.iloc[-1])

    # Momentum: last N candles direction
    price_change = float(close.iloc[-1] - close.iloc[-momentum_window])
    momentum_up = price_change > 0

    # RSI (manual calculation to avoid import issues with legacy src/)
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=rsi_period).mean()
    rs = gain / loss.replace(0, float("nan"))
    rsi_val = float(100 - (100 / (1 + rs.iloc[-1]))) if pd.notna(rs.iloc[-1]) else 50.0

    # EMA spread for confidence
    spread = abs(float(ema_f.iloc[-1] - ema_s.iloc[-1]))
    spread_prev = abs(float(ema_f.iloc[-2] - ema_s.iloc[-2]))
    spread_expanding = spread > spread_prev

    # Signal logic
    if trend_up and momentum_up:
        direction = "up"
    elif not trend_up and not momentum_up:
        direction = "down"
    else:
        now_utc = datetime.now(tz=UTC).strftime("%H:%M")
        return Signal(
            should_bet=False,
            direction="",
            streak_length=0,
            streak_direction="",
            confidence=0,
            reason=(
                f"Mixed signal at {now_utc}: "
                f"trend={'up' if trend_up else 'down'}, "
                f"momentum={'up' if momentum_up else 'down'}"
            ),
        )

    # Confidence: base 0.55, boosted when EMA spread expanding + RSI confirms
    confidence = 0.55
    if spread_expanding:
        confidence += 0.05
    if direction == "up" and 40 < rsi_val < 65:
        confidence += 0.05
    elif direction == "down" and 35 < rsi_val < 60:
        confidence += 0.05

    price = float(close.iloc[-1])
    ema_f_val = float(ema_f.iloc[-1])
    ema_s_val = float(ema_s.iloc[-1])

    return Signal(
        should_bet=True,
        direction=direction,
        streak_length=0,
        streak_direction="",
        confidence=confidence,
        reason=(
            f"Momentum {direction.upper()} | "
            f"BTC ${price:,.0f} | "
            f"EMA {ema_f_val:,.0f}/{ema_s_val:,.0f} | "
            f"RSI {rsi_val:.0f} | "
            f"conf {confidence:.0%}"
        ),
    )
