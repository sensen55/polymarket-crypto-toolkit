#!/usr/bin/env python3
"""
Wallet Trade Analyzer — Reverse-engineer a Polymarket wallet's trading strategy.

Fetches all trades for a given wallet address from the Polymarket Data API,
filters for BTC 5-min markets, and compares the trading pattern against the
streak reversal strategy implemented in this repository.

Usage:
    python scripts/analyze_wallet.py 0x5924ca480d8b08cd5f3e5811fa378c4082475af6
    python scripts/analyze_wallet.py 0x5924... --days 14
    python scripts/analyze_wallet.py 0x5924... --export csv
    python scripts/analyze_wallet.py 0x5924... --verbose
"""

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import UTC, datetime

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# ─── Constants ───────────────────────────────────────────────────────────────

DATA_API = "https://data-api.polymarket.com"
GAMMA_API = "https://gamma-api.polymarket.com"
PAGE_LIMIT = 500  # max per request

# Streak reversal rates from this repo's strategy (src/strategies/streak.py)
REPO_REVERSAL_RATES = {2: 0.540, 3: 0.579, 4: 0.667, 5: 0.824}
DEFAULT_TRIGGER = 4


# ─── Data types ──────────────────────────────────────────────────────────────


@dataclass
class WalletTrade:
    timestamp: int
    market_slug: str
    market_title: str
    side: str  # BUY or SELL
    outcome: str  # outcome label (e.g. "Up", "Down")
    outcome_index: int
    size: float  # token size
    usdc_size: float  # USD amount
    price: float
    tx_hash: str
    condition_id: str
    event_slug: str


@dataclass
class MarketWindow:
    """A single BTC 5-min market window with resolved outcome."""

    timestamp: int
    slug: str
    outcome: str | None  # "up" or "down"


@dataclass
class AnalysisResult:
    profile: dict
    total_trades: int
    btc_5m_trades: list[WalletTrade]
    non_btc_trades: int
    streak_alignment: list[dict]
    stats: dict = field(default_factory=dict)


# ─── HTTP session ────────────────────────────────────────────────────────────


def create_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(total=3, backoff_factor=0.3, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=retry)
    session.mount("https://", adapter)
    session.headers.update(
        {
            "User-Agent": "PolymarketAnalyzer/1.0",
            "Accept": "application/json",
        }
    )
    return session


# ─── API calls ───────────────────────────────────────────────────────────────


def resolve_profile(session: requests.Session, address: str) -> dict:
    """Resolve wallet profile including proxy wallet mapping."""
    resp = session.get(f"{GAMMA_API}/public-profile", params={"address": address}, timeout=10)
    if resp.status_code == 200:
        data = resp.json()
        if data:
            return data
    # Try as-is (might already be a proxy wallet)
    return {"address": address, "proxyWallet": address}


def fetch_all_activity(
    session: requests.Session,
    user: str,
    activity_type: str = "TRADE",
    start_ts: int | None = None,
    end_ts: int | None = None,
) -> list[dict]:
    """Fetch all activity records with pagination."""
    all_records = []
    offset = 0

    while True:
        params = {
            "user": user,
            "type": activity_type,
            "limit": PAGE_LIMIT,
            "offset": offset,
            "sortBy": "TIMESTAMP",
            "sortDirection": "ASC",
        }
        if start_ts:
            params["start"] = start_ts
        if end_ts:
            params["end"] = end_ts

        resp = session.get(f"{DATA_API}/activity", params=params, timeout=15)
        if resp.status_code != 200:
            print(f"[warn] Activity API returned {resp.status_code} at offset {offset}")
            break

        data = resp.json()
        if not data or not isinstance(data, list):
            break

        all_records.extend(data)
        print(f"  Fetched {len(all_records)} trade records...", end="\r")

        if len(data) < PAGE_LIMIT:
            break  # last page

        offset += PAGE_LIMIT
        time.sleep(0.2)  # rate limit courtesy

    print(f"  Fetched {len(all_records)} trade records total.")
    return all_records


def fetch_market_outcome(session: requests.Session, timestamp: int) -> str | None:
    """Fetch the resolved outcome of a BTC 5-min market."""
    slug = f"btc-updown-5m-{timestamp}"
    try:
        resp = session.get(
            f"{GAMMA_API}/events",
            params={"slug": slug},
            timeout=10,
        )
        if resp.status_code != 200:
            return None
        data = resp.json()
        if not data:
            return None
        event = data[0]
        markets = event.get("markets", [])
        if not markets:
            return None
        m = markets[0]
        prices = json.loads(m.get("outcomePrices", "[0.5, 0.5]"))
        up_price = float(prices[0]) if prices else 0.5
        down_price = float(prices[1]) if len(prices) > 1 else 0.5
        is_closed = m.get("closed", False)
        uma_status = m.get("umaResolutionStatus", "")
        is_resolved = uma_status == "resolved"
        if is_closed and (is_resolved or up_price > 0.99 or down_price > 0.99):
            if up_price > 0.99:
                return "up"
            elif down_price > 0.99:
                return "down"
    except Exception:
        pass
    return None


# ─── Parsing ─────────────────────────────────────────────────────────────────


def is_btc_5m_market(record: dict) -> bool:
    """Check if a trade is for a BTC 5-min up/down market."""
    slug = record.get("eventSlug", "") or record.get("slug", "")
    title = record.get("title", "")
    return "btc-updown-5m" in slug.lower() or "btc" in title.lower() and "5 min" in title.lower()


def parse_trade(record: dict) -> WalletTrade:
    return WalletTrade(
        timestamp=record.get("timestamp", 0),
        market_slug=record.get("eventSlug", "") or record.get("slug", ""),
        market_title=record.get("title", ""),
        side=record.get("side", ""),
        outcome=record.get("outcome", ""),
        outcome_index=record.get("outcomeIndex", -1),
        size=float(record.get("size", 0)),
        usdc_size=float(record.get("usdcSize", 0) or 0),
        price=float(record.get("price", 0)),
        tx_hash=record.get("transactionHash", ""),
        condition_id=record.get("conditionId", ""),
        event_slug=record.get("eventSlug", ""),
    )


def extract_market_timestamp(slug: str) -> int | None:
    """Extract the unix timestamp from a BTC 5-min market slug."""
    # Format: btc-updown-5m-{timestamp}
    parts = slug.split("-")
    if len(parts) >= 4:
        try:
            return int(parts[-1])
        except ValueError:
            pass
    return None


# ─── Strategy comparison ─────────────────────────────────────────────────────


def detect_streak(outcomes: list[str]) -> tuple[int, str]:
    """Detect streak at end of outcomes list (mirrors src/strategies/streak.py)."""
    if not outcomes:
        return 0, ""
    current = outcomes[-1]
    streak = 1
    for i in range(len(outcomes) - 2, -1, -1):
        if outcomes[i] == current:
            streak += 1
        else:
            break
    return streak, current


def would_streak_strategy_bet(prior_outcomes: list[str], trigger: int = DEFAULT_TRIGGER) -> dict | None:
    """Check if the streak strategy would have placed a bet given prior outcomes."""
    streak_len, streak_dir = detect_streak(prior_outcomes)
    if streak_len < trigger or not streak_dir:
        return None
    bet_direction = "down" if streak_dir == "up" else "up"
    confidence = REPO_REVERSAL_RATES.get(min(streak_len, 5), REPO_REVERSAL_RATES[5])
    return {
        "bet_direction": bet_direction,
        "streak_length": streak_len,
        "streak_direction": streak_dir,
        "confidence": confidence,
    }


def infer_bet_direction(trade: WalletTrade) -> str | None:
    """Infer which direction (up/down) the trader is betting on."""
    outcome_lower = trade.outcome.lower() if trade.outcome else ""
    side = trade.side.upper()

    # BUY "Up" = betting on up, BUY "Down" = betting on down
    # SELL "Up" = betting against up (= betting on down)
    if side == "BUY":
        if "up" in outcome_lower:
            return "up"
        elif "down" in outcome_lower:
            return "down"
    elif side == "SELL":
        if "up" in outcome_lower:
            return "down"
        elif "down" in outcome_lower:
            return "up"
    return None


# ─── Analysis ────────────────────────────────────────────────────────────────


def analyze_wallet(
    address: str,
    days: int = 14,
    trigger: int = DEFAULT_TRIGGER,
    verbose: bool = False,
) -> AnalysisResult:
    session = create_session()

    # Step 1: Resolve profile
    print(f"\n[1/5] Resolving profile for {address}...")
    profile = resolve_profile(session, address)
    proxy_wallet = profile.get("proxyWallet", address)
    pseudonym = profile.get("pseudonym", "unknown")
    print(f"  Pseudonym: {pseudonym}")
    print(f"  Proxy wallet: {proxy_wallet}")

    # Use both addresses in case one is the proxy and one is the base
    addresses_to_try = list({address.lower(), proxy_wallet.lower()})

    # Step 2: Fetch trade activity
    now = int(time.time())
    start_ts = now - (days * 86400)
    print(f"\n[2/5] Fetching trade activity (last {days} days)...")

    all_raw = []
    for addr in addresses_to_try:
        records = fetch_all_activity(session, addr, start_ts=start_ts)
        all_raw.extend(records)

    # Deduplicate by transaction hash
    seen_tx = set()
    unique_raw = []
    for r in all_raw:
        tx = r.get("transactionHash", "")
        key = (tx, r.get("outcomeIndex", ""), r.get("timestamp", ""))
        if key not in seen_tx:
            seen_tx.add(key)
            unique_raw.append(r)

    print(f"  Total unique trade records: {len(unique_raw)}")

    # Step 3: Filter and parse BTC 5-min trades
    print("\n[3/5] Filtering BTC 5-min market trades...")
    btc_trades = []
    non_btc_count = 0
    for r in unique_raw:
        if is_btc_5m_market(r):
            btc_trades.append(parse_trade(r))
        else:
            non_btc_count += 1

    btc_trades.sort(key=lambda t: t.timestamp)
    print(f"  BTC 5-min trades: {len(btc_trades)}")
    print(f"  Other market trades: {non_btc_count}")

    if not btc_trades:
        print("\n  No BTC 5-min trades found. The wallet may trade different markets.")
        return AnalysisResult(
            profile=profile,
            total_trades=len(unique_raw),
            btc_5m_trades=[],
            non_btc_trades=non_btc_count,
            streak_alignment=[],
        )

    # Step 4: Fetch market outcomes for context windows
    print("\n[4/5] Fetching market outcomes for streak analysis...")
    # Group trades by market timestamp
    market_timestamps = set()
    for t in btc_trades:
        mts = extract_market_timestamp(t.market_slug or t.event_slug)
        if mts:
            market_timestamps.add(mts)

    # Also fetch surrounding markets for streak context
    if market_timestamps:
        min_ts = min(market_timestamps)
        max_ts = max(market_timestamps)
        # Fetch 20 markets before the first trade for streak context
        context_start = min_ts - (20 * 300)
        all_ts_to_fetch = set()
        ts = context_start
        while ts <= max_ts:
            all_ts_to_fetch.add(ts)
            ts += 300

    outcomes_cache: dict[int, str | None] = {}
    total_to_fetch = len(all_ts_to_fetch)
    fetched = 0
    for ts in sorted(all_ts_to_fetch):
        outcome = fetch_market_outcome(session, ts)
        outcomes_cache[ts] = outcome
        fetched += 1
        if fetched % 20 == 0:
            print(f"  Fetched {fetched}/{total_to_fetch} market outcomes...", end="\r")
        time.sleep(0.05)  # rate limit
    print(f"  Fetched {fetched}/{total_to_fetch} market outcomes total.")

    resolved_count = sum(1 for v in outcomes_cache.values() if v)
    print(f"  Resolved outcomes: {resolved_count}/{len(outcomes_cache)}")

    # Step 5: Compare each trade against streak strategy
    print("\n[5/5] Analyzing trading pattern vs streak reversal strategy...")
    streak_alignment = []

    for trade in btc_trades:
        mts = extract_market_timestamp(trade.market_slug or trade.event_slug)
        if not mts:
            continue

        # Build prior outcomes (markets before this one)
        prior_ts_list = sorted(ts for ts in outcomes_cache if ts < mts and outcomes_cache[ts])
        prior_outcomes = [outcomes_cache[ts] for ts in prior_ts_list[-10:]]  # last 10

        # What would the streak strategy recommend?
        strategy_signal = would_streak_strategy_bet(prior_outcomes, trigger)

        # What did the wallet actually bet?
        actual_bet_dir = infer_bet_direction(trade)

        # What was the actual outcome?
        actual_outcome = outcomes_cache.get(mts)

        # Did the trade win?
        trade_won = None
        if actual_bet_dir and actual_outcome:
            trade_won = actual_bet_dir == actual_outcome

        # Streak at time of trade
        streak_len, streak_dir = detect_streak(prior_outcomes) if prior_outcomes else (0, "")

        entry = {
            "market_ts": mts,
            "market_slug": trade.market_slug or trade.event_slug,
            "trade_time": trade.timestamp,
            "trade_side": trade.side,
            "trade_outcome_label": trade.outcome,
            "trade_price": trade.price,
            "trade_usdc": trade.usdc_size,
            "inferred_bet": actual_bet_dir,
            "actual_outcome": actual_outcome,
            "trade_won": trade_won,
            "streak_before": streak_len,
            "streak_dir_before": streak_dir,
            "strategy_would_bet": strategy_signal is not None,
            "strategy_bet_dir": strategy_signal["bet_direction"] if strategy_signal else None,
            "strategy_agrees": (
                strategy_signal["bet_direction"] == actual_bet_dir if strategy_signal and actual_bet_dir else None
            ),
        }
        streak_alignment.append(entry)

        if verbose:
            dt = datetime.fromtimestamp(trade.timestamp, tz=UTC).strftime("%Y-%m-%d %H:%M")
            streak_info = f"streak={streak_len}x{streak_dir}" if streak_len > 0 else "no streak"
            strategy_info = f"strategy={'YES' if strategy_signal else 'NO'}"
            agree_info = ""
            if strategy_signal and actual_bet_dir:
                agree_info = f" {'AGREE' if strategy_signal['bet_direction'] == actual_bet_dir else 'DISAGREE'}"
            win_info = f" -> {'WIN' if trade_won else 'LOSS' if trade_won is False else '?'}"
            print(
                f"  {dt} | bet={actual_bet_dir or '?':>4} @ ${trade.usdc_size:>6.1f} | "
                f"{streak_info:>15} | {strategy_info}{agree_info}{win_info}"
            )

    # Compute statistics
    stats = compute_stats(streak_alignment, trigger)

    return AnalysisResult(
        profile=profile,
        total_trades=len(unique_raw),
        btc_5m_trades=btc_trades,
        non_btc_trades=non_btc_count,
        streak_alignment=streak_alignment,
        stats=stats,
    )


def compute_stats(alignment: list[dict], trigger: int) -> dict:
    """Compute summary statistics from the alignment analysis."""
    if not alignment:
        return {}

    total = len(alignment)
    wins = sum(1 for a in alignment if a["trade_won"] is True)
    losses = sum(1 for a in alignment if a["trade_won"] is False)
    unknown = sum(1 for a in alignment if a["trade_won"] is None)
    settled = wins + losses

    # Strategy alignment
    strategy_triggered = [a for a in alignment if a["strategy_would_bet"]]
    strategy_agrees = sum(1 for a in strategy_triggered if a["strategy_agrees"] is True)
    strategy_disagrees = sum(1 for a in strategy_triggered if a["strategy_agrees"] is False)

    # Wallet bets WITHOUT streak trigger (trading when strategy says no)
    no_trigger_trades = [a for a in alignment if not a["strategy_would_bet"]]
    no_trigger_wins = sum(1 for a in no_trigger_trades if a["trade_won"] is True)
    no_trigger_losses = sum(1 for a in no_trigger_trades if a["trade_won"] is False)

    # Wallet bets WITH streak trigger (trading when strategy says yes)
    with_trigger_trades = [a for a in alignment if a["strategy_would_bet"]]
    with_trigger_wins = sum(1 for a in with_trigger_trades if a["trade_won"] is True)
    with_trigger_losses = sum(1 for a in with_trigger_trades if a["trade_won"] is False)

    # Bet direction distribution
    bet_dirs = Counter(a["inferred_bet"] for a in alignment if a["inferred_bet"])

    # Streak length distribution at time of trade
    streak_lens = Counter(a["streak_before"] for a in alignment)

    # Size statistics
    sizes = [a["trade_usdc"] for a in alignment if a["trade_usdc"] > 0]
    avg_size = sum(sizes) / len(sizes) if sizes else 0
    min_size = min(sizes) if sizes else 0
    max_size = max(sizes) if sizes else 0

    # Price statistics
    prices = [a["trade_price"] for a in alignment if a["trade_price"] > 0]
    avg_price = sum(prices) / len(prices) if prices else 0

    # Timing analysis (seconds into the 5-min window)
    entry_offsets = []
    for a in alignment:
        mts = a["market_ts"]
        trade_ts = a["trade_time"]
        if mts and trade_ts:
            offset = trade_ts - mts
            if 0 <= offset <= 300:
                entry_offsets.append(offset)
    avg_entry_offset = sum(entry_offsets) / len(entry_offsets) if entry_offsets else 0

    # Win rate by streak length
    win_rate_by_streak = {}
    for streak_len in sorted(streak_lens.keys()):
        streak_trades = [a for a in alignment if a["streak_before"] == streak_len]
        s_wins = sum(1 for a in streak_trades if a["trade_won"] is True)
        s_losses = sum(1 for a in streak_trades if a["trade_won"] is False)
        s_settled = s_wins + s_losses
        win_rate_by_streak[streak_len] = {
            "count": len(streak_trades),
            "wins": s_wins,
            "losses": s_losses,
            "win_rate": s_wins / s_settled * 100 if s_settled > 0 else 0,
        }

    # ── Hourly win rate (UTC) ──
    hourly_stats: dict[int, dict] = {}
    for a in alignment:
        if a["trade_won"] is None:
            continue
        hour = datetime.fromtimestamp(a["trade_time"], tz=UTC).hour
        if hour not in hourly_stats:
            hourly_stats[hour] = {"wins": 0, "losses": 0}
        if a["trade_won"]:
            hourly_stats[hour]["wins"] += 1
        else:
            hourly_stats[hour]["losses"] += 1

    # ── Bet size buckets vs win rate ──
    size_buckets = {"$0-1": (0, 1), "$1-5": (1, 5), "$5-10": (5, 10), "$10-50": (10, 50), "$50+": (50, 1e9)}
    size_bucket_stats: dict[str, dict] = {}
    for label, (lo, hi) in size_buckets.items():
        bucket_trades = [a for a in alignment if lo <= a["trade_usdc"] < hi and a["trade_won"] is not None]
        bw = sum(1 for a in bucket_trades if a["trade_won"])
        bl = len(bucket_trades) - bw
        if bucket_trades:
            size_bucket_stats[label] = {
                "count": len(bucket_trades),
                "wins": bw,
                "losses": bl,
                "win_rate": bw / len(bucket_trades) * 100,
                "avg_size": sum(a["trade_usdc"] for a in bucket_trades) / len(bucket_trades),
            }

    # ── Entry timing buckets vs win rate ──
    timing_buckets = {
        "0-60s": (0, 60), "60-120s": (60, 120), "120-180s": (120, 180),
        "180-240s": (180, 240), "240-300s": (240, 300),
    }
    timing_bucket_stats: dict[str, dict] = {}
    for label, (lo, hi) in timing_buckets.items():
        bucket_trades = [
            a for a in alignment
            if a["trade_won"] is not None
            and lo <= (a["trade_time"] - a["market_ts"]) < hi
            and 0 <= (a["trade_time"] - a["market_ts"]) <= 300
        ]
        tw = sum(1 for a in bucket_trades if a["trade_won"])
        tl = len(bucket_trades) - tw
        if bucket_trades:
            timing_bucket_stats[label] = {
                "count": len(bucket_trades), "wins": tw, "losses": tl,
                "win_rate": tw / len(bucket_trades) * 100,
            }

    # ── Estimated P&L ──
    total_pnl = 0.0
    gross_profit = 0.0
    gross_loss = 0.0
    for a in alignment:
        if a["trade_won"] is None:
            continue
        cost = a["trade_usdc"]
        entry_price = a["trade_price"]
        if entry_price <= 0 or cost <= 0:
            continue
        tokens = cost / entry_price
        if a["trade_won"]:
            profit = tokens - cost  # payout = tokens * 1.0, cost = tokens * price
            total_pnl += profit
            gross_profit += profit
        else:
            total_pnl -= cost
            gross_loss += cost

    # ── Post-win / post-loss behavior ──
    post_win_stats = {"wins": 0, "losses": 0}
    post_loss_stats = {"wins": 0, "losses": 0}
    for i in range(1, len(alignment)):
        prev = alignment[i - 1]["trade_won"]
        curr = alignment[i]["trade_won"]
        if prev is None or curr is None:
            continue
        if prev is True:
            if curr:
                post_win_stats["wins"] += 1
            else:
                post_win_stats["losses"] += 1
        else:
            if curr:
                post_loss_stats["wins"] += 1
            else:
                post_loss_stats["losses"] += 1

    # ── Wallet's own win/loss streaks ──
    wallet_streaks: list[tuple[str, int]] = []  # (type, length)
    current_streak_type = None
    current_streak_len = 0
    for a in alignment:
        if a["trade_won"] is None:
            continue
        result_type = "W" if a["trade_won"] else "L"
        if result_type == current_streak_type:
            current_streak_len += 1
        else:
            if current_streak_type is not None:
                wallet_streaks.append((current_streak_type, current_streak_len))
            current_streak_type = result_type
            current_streak_len = 1
    if current_streak_type is not None:
        wallet_streaks.append((current_streak_type, current_streak_len))

    win_streaks = [length for stype, length in wallet_streaks if stype == "W"]
    loss_streaks = [length for stype, length in wallet_streaks if stype == "L"]

    return {
        "total_btc_trades": total,
        "wins": wins,
        "losses": losses,
        "unsettled": unknown,
        "settled": settled,
        "win_rate": wins / settled * 100 if settled > 0 else 0,
        "strategy_triggered_count": len(strategy_triggered),
        "strategy_agrees": strategy_agrees,
        "strategy_disagrees": strategy_disagrees,
        "agreement_rate": (strategy_agrees / len(strategy_triggered) * 100 if strategy_triggered else 0),
        "no_trigger_count": len(no_trigger_trades),
        "no_trigger_wins": no_trigger_wins,
        "no_trigger_losses": no_trigger_losses,
        "no_trigger_win_rate": (
            no_trigger_wins / (no_trigger_wins + no_trigger_losses) * 100
            if (no_trigger_wins + no_trigger_losses) > 0
            else 0
        ),
        "with_trigger_count": len(with_trigger_trades),
        "with_trigger_wins": with_trigger_wins,
        "with_trigger_losses": with_trigger_losses,
        "with_trigger_win_rate": (
            with_trigger_wins / (with_trigger_wins + with_trigger_losses) * 100
            if (with_trigger_wins + with_trigger_losses) > 0
            else 0
        ),
        "bet_directions": dict(bet_dirs),
        "streak_distribution": dict(streak_lens),
        "avg_bet_size_usd": avg_size,
        "min_bet_size_usd": min_size,
        "max_bet_size_usd": max_size,
        "avg_entry_price": avg_price,
        "avg_entry_seconds_into_window": avg_entry_offset,
        "win_rate_by_streak": win_rate_by_streak,
        # New deep analysis fields
        "hourly_stats": hourly_stats,
        "size_bucket_stats": size_bucket_stats,
        "timing_bucket_stats": timing_bucket_stats,
        "estimated_pnl": total_pnl,
        "gross_profit": gross_profit,
        "gross_loss": gross_loss,
        "post_win_stats": post_win_stats,
        "post_loss_stats": post_loss_stats,
        "wallet_win_streaks": win_streaks,
        "wallet_loss_streaks": loss_streaks,
    }


# ─── Output ──────────────────────────────────────────────────────────────────


def print_report(result: AnalysisResult, trigger: int):
    s = result.stats
    if not s:
        print("\nNo BTC 5-min trades to analyze.")
        return

    print("\n" + "=" * 70)
    print("WALLET TRADE ANALYSIS REPORT")
    print("=" * 70)

    pseudo = result.profile.get("pseudonym", "unknown")
    proxy = result.profile.get("proxyWallet", "?")
    print(f"\n  Profile:          {pseudo}")
    print(f"  Proxy wallet:     {proxy}")
    print(f"  Total trades:     {result.total_trades}")
    print(f"  BTC 5-min trades: {len(result.btc_5m_trades)}")
    print(f"  Other markets:    {result.non_btc_trades}")

    print("\n--- Performance ---")
    print(f"  Settled:   {s['settled']} ({s['wins']}W / {s['losses']}L)")
    print(f"  Win rate:  {s['win_rate']:.1f}%")
    print(f"  Unsettled: {s['unsettled']}")

    print("\n--- Bet Sizing ---")
    print(f"  Average: ${s['avg_bet_size_usd']:.2f}")
    print(f"  Min:     ${s['min_bet_size_usd']:.2f}")
    print(f"  Max:     ${s['max_bet_size_usd']:.2f}")
    print(f"  Avg entry price: {s['avg_entry_price']:.3f}")

    print("\n--- Timing ---")
    print(f"  Avg entry: {s['avg_entry_seconds_into_window']:.0f}s into 5-min window")

    print("\n--- Bet Directions ---")
    for direction, count in sorted(s["bet_directions"].items()):
        print(f"  {direction or 'unknown':>5}: {count}")

    print(f"\n--- Streak Reversal Strategy Comparison (trigger={trigger}) ---")
    print(f"  Trades where strategy would also bet:  {s['strategy_triggered_count']}")
    if s["strategy_triggered_count"] > 0:
        print(f"    Same direction (agrees):    {s['strategy_agrees']}")
        print(f"    Opposite direction:         {s['strategy_disagrees']}")
        print(f"    Agreement rate:             {s['agreement_rate']:.1f}%")
    print(f"  Trades where strategy would NOT bet:   {s['no_trigger_count']}")

    if s["with_trigger_count"] > 0:
        print(
            f"\n  Win rate WHEN strategy triggered: {s['with_trigger_win_rate']:.1f}% "
            f"({s['with_trigger_wins']}W/{s['with_trigger_losses']}L)"
        )
    if s["no_trigger_count"] > 0:
        settled_no = s["no_trigger_wins"] + s["no_trigger_losses"]
        if settled_no > 0:
            print(
                f"  Win rate when NO streak trigger: {s['no_trigger_win_rate']:.1f}% "
                f"({s['no_trigger_wins']}W/{s['no_trigger_losses']}L)"
            )

    print("\n--- Win Rate by Streak Length ---")
    print(f"  {'Streak':>6}  {'Count':>5}  {'Win Rate':>8}  {'W/L':>7}  {'Repo Rate':>9}")
    for streak_len, data in sorted(s["win_rate_by_streak"].items()):
        repo_rate = REPO_REVERSAL_RATES.get(min(streak_len, 5), 0)
        repo_str = f"{repo_rate:.1%}" if repo_rate > 0 else "n/a"
        print(
            f"  {streak_len:>6}  {data['count']:>5}  {data['win_rate']:>7.1f}%  "
            f"{data['wins']}W/{data['losses']}L  {repo_str:>9}"
        )

    # ── Estimated P&L ──
    print("\n--- Estimated P&L ---")
    pnl = s.get("estimated_pnl", 0)
    gp = s.get("gross_profit", 0)
    gl = s.get("gross_loss", 0)
    pnl_sign = "+" if pnl >= 0 else ""
    print(f"  Net P&L:      {pnl_sign}${pnl:,.2f}")
    print(f"  Gross profit: +${gp:,.2f}")
    print(f"  Gross loss:   -${gl:,.2f}")
    if s["settled"] > 0:
        print(f"  Avg P&L/trade: {pnl_sign}${pnl / s['settled']:.4f}")

    # ── Hourly Win Rate ──
    hourly = s.get("hourly_stats", {})
    if hourly:
        print("\n--- Win Rate by Hour (UTC) ---")
        print(f"  {'Hour':>6}  {'Count':>5}  {'Win Rate':>8}  {'W/L':>10}  {'Bar'}")
        for h in range(24):
            if h not in hourly:
                continue
            hw = hourly[h]["wins"]
            hl = hourly[h]["losses"]
            ht = hw + hl
            if ht == 0:
                continue
            wr = hw / ht * 100
            bar = "#" * int(wr / 2)
            print(f"  {h:>4}:00  {ht:>5}  {wr:>7.1f}%  {hw}W/{hl}L  {bar}")

    # ── Bet Size vs Win Rate ──
    sbs = s.get("size_bucket_stats", {})
    if sbs:
        print("\n--- Win Rate by Bet Size ---")
        print(f"  {'Bucket':>8}  {'Count':>5}  {'Win Rate':>8}  {'W/L':>10}  {'Avg Size':>8}")
        for label in ["$0-1", "$1-5", "$5-10", "$10-50", "$50+"]:
            if label not in sbs:
                continue
            b = sbs[label]
            print(
                f"  {label:>8}  {b['count']:>5}  {b['win_rate']:>7.1f}%"
                f"  {b['wins']}W/{b['losses']}L  ${b['avg_size']:>7.2f}"
            )

    # ── Entry Timing vs Win Rate ──
    tbs = s.get("timing_bucket_stats", {})
    if tbs:
        print("\n--- Win Rate by Entry Timing ---")
        print(f"  {'Window':>8}  {'Count':>5}  {'Win Rate':>8}  {'W/L':>10}")
        for label in ["0-60s", "60-120s", "120-180s", "180-240s", "240-300s"]:
            if label not in tbs:
                continue
            t = tbs[label]
            print(f"  {label:>8}  {t['count']:>5}  {t['win_rate']:>7.1f}%  {t['wins']}W/{t['losses']}L")

    # ── Post-Win / Post-Loss Behavior ──
    pw = s.get("post_win_stats", {})
    pl = s.get("post_loss_stats", {})
    if pw.get("wins", 0) + pw.get("losses", 0) > 0 or pl.get("wins", 0) + pl.get("losses", 0) > 0:
        print("\n--- Behavior After Win/Loss ---")
        pw_total = pw["wins"] + pw["losses"]
        pl_total = pl["wins"] + pl["losses"]
        if pw_total > 0:
            print(f"  After WIN:  {pw['wins']}W/{pw['losses']}L  ({pw['wins'] / pw_total * 100:.1f}% win rate)")
        if pl_total > 0:
            print(f"  After LOSS: {pl['wins']}W/{pl['losses']}L  ({pl['wins'] / pl_total * 100:.1f}% win rate)")

    # ── Wallet's Own Streaks ──
    ws = s.get("wallet_win_streaks", [])
    ls_ = s.get("wallet_loss_streaks", [])
    if ws or ls_:
        print("\n--- Wallet Win/Loss Streaks ---")
        if ws:
            print(f"  Win streaks:  max={max(ws)}, avg={sum(ws)/len(ws):.1f}, count={len(ws)}")
            ws_dist = Counter(ws)
            for length in sorted(ws_dist):
                if length >= 3:
                    print(f"    {length}+ wins in a row: {ws_dist[length]}x")
        if ls_:
            print(f"  Loss streaks: max={max(ls_)}, avg={sum(ls_)/len(ls_):.1f}, count={len(ls_)}")
            ls_dist = Counter(ls_)
            for length in sorted(ls_dist):
                if length >= 3:
                    print(f"    {length}+ losses in a row: {ls_dist[length]}x")

    # Verdict
    print("\n--- Verdict ---")
    if s["agreement_rate"] > 70 and s["strategy_triggered_count"] > 10:
        print("  HIGH correlation with streak reversal strategy.")
        print("  This wallet appears to use a similar approach to this repo's strategy.")
    elif s["agreement_rate"] > 40 and s["strategy_triggered_count"] > 5:
        print("  MODERATE correlation with streak reversal strategy.")
        print("  The wallet may use streak analysis as one factor among others.")
    elif s["no_trigger_count"] > s["strategy_triggered_count"] * 2:
        print("  LOW correlation with streak reversal strategy.")
        print("  Most trades happen when there is no significant streak.")
        print("  The wallet likely uses a DIFFERENT strategy (e.g., price-based, timing-based).")
    else:
        print("  INCONCLUSIVE — insufficient data for a confident assessment.")

    print("=" * 70 + "\n")


def export_results(result: AnalysisResult, fmt: str, output_path: str | None = None):
    if fmt == "json":
        path = output_path or "wallet_analysis.json"
        export = {
            "profile": result.profile,
            "stats": result.stats,
            "trades": [
                {
                    "timestamp": t.timestamp,
                    "slug": t.market_slug,
                    "side": t.side,
                    "outcome": t.outcome,
                    "size": t.size,
                    "usdc_size": t.usdc_size,
                    "price": t.price,
                }
                for t in result.btc_5m_trades
            ],
            "alignment": result.streak_alignment,
        }
        with open(path, "w") as f:
            json.dump(export, f, indent=2, default=str)
        print(f"Exported JSON to {path}")

    elif fmt == "csv":
        path = output_path or "wallet_analysis.csv"
        import csv

        with open(path, "w", newline="") as f:
            if result.streak_alignment:
                writer = csv.DictWriter(f, fieldnames=result.streak_alignment[0].keys())
                writer.writeheader()
                writer.writerows(result.streak_alignment)
        print(f"Exported CSV to {path}")


# ─── CLI ─────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="Analyze a Polymarket wallet's BTC 5-min trading strategy")
    parser.add_argument("address", help="Wallet address (0x...)")
    parser.add_argument("--days", type=int, default=14, help="Days of history to fetch (default: 14)")
    parser.add_argument(
        "--trigger",
        type=int,
        default=DEFAULT_TRIGGER,
        help=f"Streak trigger for comparison (default: {DEFAULT_TRIGGER})",
    )
    parser.add_argument("--export", choices=["json", "csv"], help="Export results")
    parser.add_argument("--output", type=str, help="Output file path for export")
    parser.add_argument("--verbose", "-v", action="store_true", help="Show each trade detail")
    args = parser.parse_args()

    if not args.address.startswith("0x") or len(args.address) != 42:
        print("Error: Invalid Ethereum address format (expected 0x + 40 hex chars)")
        sys.exit(1)

    result = analyze_wallet(
        address=args.address,
        days=args.days,
        trigger=args.trigger,
        verbose=args.verbose,
    )

    print_report(result, args.trigger)

    if args.export:
        export_results(result, args.export, args.output)


if __name__ == "__main__":
    main()
