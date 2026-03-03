"""
BTC Momentum Strategy — follow short-term BTC price trends.

Inspired by analysis of high-frequency Polymarket wallets that:
- Follow BTC price momentum (same direction for extended runs)
- Size bets based on signal confidence (bigger bets = higher win rate)
- Enter 60-120s into the 5-min window after observing initial price movement
- Use EMA trend + recent price action rather than market outcome history

The strategy uses EMA crossover for trend direction and recent candle
momentum for confirmation, producing a signal on every evaluation.
"""

from __future__ import annotations

from typing import Any, cast

import pandas as pd
from polymarket_algo.indicators import ema, rsi


class BTCMomentumStrategy:
    name = "btc_momentum"
    description = "BTC price momentum strategy with confidence-based sizing"
    timeframe = "5m"

    @property
    def default_params(self) -> dict[str, Any]:
        return {
            "ema_fast": 8,
            "ema_slow": 21,
            "rsi_period": 14,
            "momentum_window": 3,
            "base_size": 5.0,
            "strong_size": 15.0,
        }

    @property
    def param_grid(self) -> dict[str, list[Any]]:
        return {
            "ema_fast": [5, 8, 12],
            "ema_slow": [13, 21, 26],
            "rsi_period": [10, 14],
            "momentum_window": [2, 3, 5],
            "base_size": [3.0, 5.0],
            "strong_size": [10.0, 15.0, 20.0],
        }

    def evaluate(self, candles: pd.DataFrame, **params: Any) -> pd.DataFrame:
        config = {**self.default_params, **params}

        close = cast(pd.Series, candles["close"])

        # Trend: EMA crossover
        ema_fast = ema(close, int(config["ema_fast"]))
        ema_slow = ema(close, int(config["ema_slow"]))
        trend_up = ema_fast > ema_slow

        # Momentum: recent price direction over N candles
        mom_window = int(config["momentum_window"])
        price_change = close - close.shift(mom_window)
        momentum_up = price_change > 0

        # RSI for confidence weighting (not filtering)
        rsi_val = rsi(close, period=int(config["rsi_period"]))

        # Signal: follow trend when momentum confirms
        signal = pd.Series(0, index=candles.index, dtype=int)
        signal.loc[trend_up & momentum_up] = 1
        signal.loc[~trend_up & ~momentum_up] = -1

        # Confidence-based sizing (key insight from wallet analysis)
        base = float(config["base_size"])
        strong = float(config["strong_size"])

        size = pd.Series(base, index=candles.index)

        # Strong signal: EMA spread is widening + RSI confirms
        ema_spread = (ema_fast - ema_slow).abs()
        ema_spread_expanding = ema_spread > ema_spread.shift(1)

        strong_long = (signal == 1) & ema_spread_expanding & rsi_val.between(40, 65)
        strong_short = (signal == -1) & ema_spread_expanding & rsi_val.between(35, 60)
        size.loc[strong_long | strong_short] = strong

        size.loc[signal == 0] = 0.0

        return pd.DataFrame({"signal": signal, "size": size}, index=candles.index)
