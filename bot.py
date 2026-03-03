#!/usr/bin/env python3
# DEPRECATED: Use polymarket_algo.* packages instead. This file exists for backward compatibility.
"""
Polymarket BTC 5-Min Streak Reversal Bot

Monitors BTC 5-min markets on Polymarket and bets against streaks.
"""

import argparse
import signal
import time
from datetime import datetime

from src.config import LOCAL_TZ, TIMEZONE_NAME, Config
from src.core.polymarket import PolymarketClient
from src.core.trader import LiveTrader, PaperTrader, TradingState
from src.strategies.streak import evaluate as streak_evaluate
from src.strategies.streak import kelly_size

# Lazy-loaded to avoid import error when not used
_momentum_evaluate = None


def _get_momentum_evaluate():
    global _momentum_evaluate
    if _momentum_evaluate is None:
        from src.strategies.momentum import evaluate as _meval

        _momentum_evaluate = _meval
    return _momentum_evaluate


running = True


def handle_signal(sig, _frame):
    global running
    print("\n[bot] Shutting down gracefully...")
    running = False


def log(msg: str):
    ts = datetime.now(LOCAL_TZ).strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")


def main():
    global running
    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    parser = argparse.ArgumentParser(
        description="Polymarket BTC 5-Min Streak Reversal Bot",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Environment Variables (.env):
  PAPER_TRADE        Paper trading mode (default: true)
  STREAK_TRIGGER     Streak length to trigger bet (default: 4)
  BET_AMOUNT         Bet amount in USD (default: 5)
  MIN_BET            Minimum bet size (default: 1)
  MAX_DAILY_BETS     Maximum bets per day (default: 50)
  MAX_DAILY_LOSS     Stop trading after this loss (default: 50)
  ENTRY_SECONDS_BEFORE  Seconds before window to enter (default: 30)
  TIMEZONE           Display timezone (default: Asia/Jakarta)
  PRIVATE_KEY        Polygon wallet private key (required for live)

Current Configuration:
  Mode:              {"PAPER" if Config.PAPER_TRADE else "LIVE"}
  Streak Trigger:    {Config.STREAK_TRIGGER}
  Bet Amount:        ${Config.BET_AMOUNT}
  Min Bet:           ${Config.MIN_BET}
  Max Daily Bets:    {Config.MAX_DAILY_BETS}
  Max Daily Loss:    ${Config.MAX_DAILY_LOSS}
  Entry Before:      {Config.ENTRY_SECONDS_BEFORE}s
  Timezone:          {TIMEZONE_NAME}

Examples:
  python bot.py --paper                    # Paper trade with defaults
  python bot.py --paper --trigger 5        # Trigger on 5-streak
  python bot.py --paper --amount 10        # Bet $10 per trade
  python bot.py --paper --bankroll 500     # Start with $500 bankroll
  python bot.py                            # Live trade (requires PRIVATE_KEY)

Related Commands:
  python history.py --stats                # View trading statistics
  python history.py --export csv           # Export trade history
  python copybot.py --help                 # Copytrade bot help
""",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help=f"Force paper trading mode (current: {Config.PAPER_TRADE})",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Force live trading mode (requires PRIVATE_KEY)",
    )
    parser.add_argument(
        "--trigger",
        type=int,
        metavar="N",
        help=f"Streak length to trigger bet (default: {Config.STREAK_TRIGGER})",
    )
    parser.add_argument(
        "--amount",
        type=float,
        metavar="USD",
        help=f"Bet amount in USD (default: {Config.BET_AMOUNT})",
    )
    parser.add_argument(
        "--bankroll",
        type=float,
        metavar="USD",
        help="Set starting bankroll (overrides saved state)",
    )
    parser.add_argument(
        "--strategy",
        choices=["streak", "momentum"],
        default="streak",
        help="Trading strategy: streak (reversal) or momentum (BTC price)",
    )
    parser.add_argument(
        "--max-bets",
        type=int,
        metavar="N",
        help=f"Maximum daily bets (default: {Config.MAX_DAILY_BETS})",
    )
    parser.add_argument(
        "--max-loss",
        type=float,
        metavar="USD",
        help=f"Stop after this daily loss (default: {Config.MAX_DAILY_LOSS})",
    )
    args = parser.parse_args()

    # Determine trading mode
    if args.live:
        paper_mode = False
    elif args.paper:
        paper_mode = True
    else:
        paper_mode = Config.PAPER_TRADE

    trigger = args.trigger or Config.STREAK_TRIGGER
    bet_amount = args.amount or Config.BET_AMOUNT
    max_daily_bets = args.max_bets or Config.MAX_DAILY_BETS
    max_daily_loss = args.max_loss or Config.MAX_DAILY_LOSS

    # Init
    client = PolymarketClient()
    state = TradingState.load()
    if args.bankroll:
        state.bankroll = args.bankroll

    if paper_mode:
        trader = PaperTrader()
        log("Paper trading mode")
    else:
        trader = LiveTrader()
        log("LIVE trading mode - Real money!")

    log(f"Config: trigger={trigger}, amount=${bet_amount}, bankroll=${state.bankroll:.2f}")
    log(f"Limits: max_bets={max_daily_bets}/day, max_loss=${max_daily_loss}")
    log(f"Timezone: {TIMEZONE_NAME}")

    strategy_name = args.strategy
    log(
        f"Strategy: {strategy_name}"
        + (f" (trigger={trigger})" if strategy_name == "streak" else " (BTC price momentum)")
    )
    log(f"Bankroll: ${state.bankroll:.2f}")
    log(f"Limits: max {Config.MAX_DAILY_BETS} bets/day, max ${Config.MAX_DAILY_LOSS} loss/day")
    log("")

    # Track what we've already bet on
    bet_timestamps: set[int] = {t.timestamp for t in state.trades}
    # Track pending trades (bet placed, waiting for resolution)
    pending: list = []

    while running:
        try:
            now = int(time.time())
            current_window = (now // 300) * 300
            seconds_into_window = now - current_window
            next_window = current_window + 300

            # === SETTLE PENDING TRADES ===
            for trade in list(pending):
                market = client.get_market(trade.timestamp)
                if market and market.closed and market.outcome:
                    state.settle_trade(trade, market.outcome)
                    emoji = "✓" if trade.pnl > 0 else "✗"
                    won = trade.direction == market.outcome
                    fee_info = f" (fee: {trade.fee_pct:.2%})" if won and trade.fee_pct > 0 else ""
                    log(
                        f"[{emoji}] Settled: {trade.direction.upper()} @ {trade.execution_price:.3f} "
                        f"-> {market.outcome.upper()} | PnL: ${trade.pnl:+.2f}{fee_info} "
                        f"| Bankroll: ${state.bankroll:.2f}"
                    )
                    pending.remove(trade)
                    state.save()

            # === CHECK IF WE CAN TRADE ===
            can_trade, reason = state.can_trade()
            if not can_trade:
                if seconds_into_window == 0:
                    log(f"⏸️  {reason}")
                time.sleep(10)
                continue

            # === DETERMINE TARGET MARKET ===
            # We want to bet on the next window, entering before it starts
            target_ts = next_window
            seconds_until_target = target_ts - now

            # Already bet on this market?
            if target_ts in bet_timestamps:
                time.sleep(5)
                continue

            # === ENTRY TIMING ===
            # Wait until we're within ENTRY_SECONDS_BEFORE of the target window
            if seconds_until_target > Config.ENTRY_SECONDS_BEFORE:
                if seconds_into_window % 60 == 0:
                    log(
                        f"⏳ Next window in {seconds_until_target}s "
                        f"(entering at T-{Config.ENTRY_SECONDS_BEFORE}s) | "
                        f"Pending: {len(pending)} trades"
                    )
                time.sleep(1)
                continue

            # === EVALUATE STRATEGY ===
            if strategy_name == "momentum":
                log("🔍 Fetching BTC price data...")
                momentum_eval = _get_momentum_evaluate()
                sig = momentum_eval()
            else:
                log("🔍 Fetching recent outcomes...")
                outcomes = client.get_recent_outcomes(count=trigger + 2)
                if len(outcomes) < trigger:
                    log(f"⚠️  Only {len(outcomes)} recent outcomes, need {trigger}")
                    bet_timestamps.add(target_ts)  # skip this window
                    time.sleep(5)
                    continue
                log(f"📊 Recent outcomes: {' → '.join(o.upper() for o in outcomes)}")
                sig = streak_evaluate(outcomes, trigger=trigger)

            if not sig.should_bet:
                log(f"🟡 No signal: {sig.reason}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # === GET TARGET MARKET ===
            market = client.get_market(target_ts)
            if not market:
                log(f"⚠️  Market not found for ts={target_ts}")
                time.sleep(5)
                continue

            if not market.accepting_orders:
                log(f"⚠️  Market not accepting orders: {market.slug}")
                bet_timestamps.add(target_ts)
                time.sleep(5)
                continue

            # === CALCULATE BET SIZE ===
            entry_price = market.up_price if sig.direction == "up" else market.down_price
            if entry_price <= 0:
                entry_price = 0.5

            odds = 1.0 / entry_price  # decimal odds
            amount = min(
                kelly_size(sig.confidence, odds, state.bankroll),
                bet_amount,
                state.bankroll * 0.1,  # never risk more than 10% of bankroll
            )
            amount = max(Config.MIN_BET, amount)

            # === PLACE BET ===
            log(f"🎯 Signal: {sig.reason}")
            trade = trader.place_bet(
                market=market,
                direction=sig.direction,
                amount=amount,
                confidence=sig.confidence,
                streak_length=sig.streak_length,
            )

            # Handle rejected orders (e.g., below minimum size)
            if trade is None:
                log("❌ Order rejected")
                bet_timestamps.add(target_ts)  # Don't retry this market
                continue

            state.record_trade(trade)
            bet_timestamps.add(target_ts)
            pending.append(trade)
            state.save()

            # === STATUS ===
            log(
                f"📈 Daily: {state.daily_bets} bets, PnL: ${state.daily_pnl:+.2f} "
                f"| Bankroll: ${state.bankroll:.2f} | Pending: {len(pending)}"
            )

            time.sleep(5)

        except KeyboardInterrupt:
            break
        except Exception as e:
            log(f"❌ Error: {e}")
            time.sleep(10)

    # Save state on exit
    state.save()
    log(f"💾 State saved. Bankroll: ${state.bankroll:.2f}")
    log(f"📊 Session: {state.daily_bets} bets, PnL: ${state.daily_pnl:+.2f}")


if __name__ == "__main__":
    main()
