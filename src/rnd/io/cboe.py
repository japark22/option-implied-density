"""Parser for Cboe's public delayed-quote JSON (cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json).

Layout (as published): {"timestamp": str, "data": {"options": [ {...}, ... ], <scalar fields>}}.
``timestamp`` is the payload generation time in UTC (it matched the fetch time
in a live check on 2026-10-09), not the quotes' as-of time - see ``asof_time``.
Option ``last_trade_time`` values are US/Eastern (live check 2026-10-09: the
latest trade was 16:14:59, the SPX options close), and the delayed feed does
not include global-trading-hours trades.
Each option has ``option`` (OCC-style symbol such as ``SPXW261009C07500000``),
``bid``, ``ask``, ``bid_size``, ``ask_size``, ``iv``, ``open_interest``, ``volume``,
``last_trade_price``, ``last_trade_time`` and greeks.

Settlement matters: root ``SPX`` (third-Friday monthlies) is AM-settled on the
Special Opening Quotation, while ``SPXW`` is PM-settled on the 4:00 pm close.
Anything compared with a 4 pm event contract must use ``SPXW`` only.
"""
from __future__ import annotations

import re
from typing import Optional

import numpy as np
import pandas as pd

from ..chain import OptionChain

SYMBOL_RE = re.compile(r"^(?P<root>[A-Z]+?)(?P<ymd>\d{6})(?P<cp>[CP])(?P<strike>\d{8})$")
NUMERIC = ["bid", "ask", "bid_size", "ask_size", "iv", "open_interest", "volume", "last_trade_price"]
ET = "America/New_York"


def parse_symbol(sym: str) -> Optional[dict]:
    m = SYMBOL_RE.match(sym.strip())
    if not m:
        return None
    return dict(root=m["root"], expiry=pd.Timestamp("20" + m["ymd"]).date(), cp=m["cp"],
                strike=int(m["strike"]) / 1000.0)


def parse_cboe_delayed(payload: dict) -> tuple[pd.DataFrame, dict]:
    """Return (options long table, metadata of scalar fields incl. Cboe timestamp)."""
    data = payload.get("data", {}) or {}
    rows = []
    for o in data.get("options", []) or []:
        p = parse_symbol(str(o.get("option", "")))
        if p is None:
            continue
        for f in NUMERIC:
            v = o.get(f)
            p[f] = float(v) if v is not None and v != "" else np.nan
        p["last_trade_time"] = o.get("last_trade_time")
        rows.append(p)
    df = pd.DataFrame(rows)
    meta = {k: v for k, v in data.items() if not isinstance(v, (list, dict))}
    meta["cboe_timestamp"] = payload.get("timestamp")
    return df, meta


def settlement_time(expiry, root: str, pm_close: str = "16:00") -> pd.Timestamp:
    """PM-settled roots expire at the close (16:00 ET, 13:00 on early-close days); AM-settled SPX at 09:30 ET."""
    t = "09:30" if root == "SPX" else pm_close
    return pd.Timestamp(f"{expiry} {t}").tz_localize(ET)


def asof_time(df: pd.DataFrame, fetched_at: pd.Timestamp) -> pd.Timestamp:
    """As-of time of a delayed payload: the latest trade it contains (read as ET), capped at fetch time.

    The feed is delayed, so the fetch time overstates how current the quotes
    are. The newest ``last_trade_time`` is a conservative proxy for the
    moment the snapshot reflects.
    """
    t = pd.to_datetime(df.get("last_trade_time"), errors="coerce").dropna()
    fetched_et = fetched_at.tz_convert(ET)
    if t.empty:
        return fetched_et
    latest = t.max()
    latest = latest.tz_localize(ET) if latest.tzinfo is None else latest.tz_convert(ET)
    return min(latest, fetched_et)


def chain_from_table(df: pd.DataFrame, expiry, quote_time: pd.Timestamp, root: str = "SPXW",
                     min_T_minutes: float = 5.0, pm_close: str = "16:00") -> OptionChain:
    """Select one root/expiry and build an OptionChain.

    ``quote_time`` must be tz-aware. T is calendar time to settlement in years.
    """
    exp = pd.Timestamp(expiry).date()
    d = df[(df["root"] == root) & (df["expiry"] == exp)]
    if d.empty:
        raise ValueError(f"no {root} options for expiry {exp}")
    settle = settlement_time(exp, root, pm_close)
    T = (settle - quote_time.tz_convert(ET)).total_seconds() / (365.0 * 24 * 3600)
    if T * 365 * 24 * 60 < min_T_minutes:
        raise ValueError(f"expiry {exp} is less than {min_T_minutes} minutes away")
    return OptionChain.from_frame(d[["strike", "cp", "bid", "ask"]], T=T, quote_time=quote_time,
                                  expiry=settle, underlying=root)
