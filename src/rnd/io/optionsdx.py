"""Loader for end-of-day option files in the OptionsDX-style wide layout.

Expected columns (brackets and whitespace are stripped, case-insensitive):
QUOTE_DATE, QUOTE_TIME_HOURS (optional), UNDERLYING_LAST (optional),
EXPIRE_DATE, STRIKE, C_BID, C_ASK, P_BID, P_ASK.

These files carry no option root, so a PM-settled weekly (SPXW) and the
AM-settled monthly (SPX) that share a third-Friday expiry cannot be told
apart. ``drop_third_fridays=True`` (default) removes those expiries rather
than risk mixing settlement styles. Verify the column names against your own
download before use; ``rename`` lets you map differences.
"""
from __future__ import annotations

import re
from typing import Iterator, Optional

import pandas as pd

from ..chain import OptionChain

ET = "America/New_York"
REQUIRED = ["QUOTE_DATE", "EXPIRE_DATE", "STRIKE", "C_BID", "C_ASK", "P_BID", "P_ASK"]


def _norm(c: str) -> str:
    return re.sub(r"[\[\]\s]", "", c).upper()


def is_third_friday(d) -> bool:
    d = pd.Timestamp(d)
    return d.weekday() == 4 and 15 <= d.day <= 21


def load_eod(path: str, rename: Optional[dict] = None) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    df.columns = [_norm(c) for c in df.columns]
    if rename:
        df = df.rename(columns={_norm(k): _norm(v) for k, v in rename.items()})
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}; got {list(df.columns)[:12]}...")
    for c in ("STRIKE", "C_BID", "C_ASK", "P_BID", "P_ASK", "UNDERLYING_LAST", "QUOTE_TIME_HOURS"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    df["QUOTE_DATE"] = pd.to_datetime(df["QUOTE_DATE"].astype(str).str.strip()).dt.date
    df["EXPIRE_DATE"] = pd.to_datetime(df["EXPIRE_DATE"].astype(str).str.strip()).dt.date
    return df


def iter_chains(df: pd.DataFrame, quote_hour: float = 16.0, drop_third_fridays: bool = True,
                max_dte: int = 10) -> Iterator[OptionChain]:
    """Yield one OptionChain per (quote date, expiry), PM settlement assumed."""
    for (qd, ed), g in df.groupby(["QUOTE_DATE", "EXPIRE_DATE"]):
        if ed <= qd or (pd.Timestamp(ed) - pd.Timestamp(qd)).days > max_dte:
            continue
        if drop_third_fridays and is_third_friday(ed):
            continue
        g = g.drop_duplicates("STRIKE", keep="last")
        hours = float(g["QUOTE_TIME_HOURS"].iloc[0]) if "QUOTE_TIME_HOURS" in g else quote_hour
        qt = (pd.Timestamp(qd) + pd.Timedelta(hours=hours)).tz_localize(ET)
        settle = pd.Timestamp(f"{ed} 16:00").tz_localize(ET)
        T = (settle - qt).total_seconds() / (365.0 * 24 * 3600)
        spot = float(g["UNDERLYING_LAST"].iloc[0]) if "UNDERLYING_LAST" in g else None
        yield OptionChain(strikes=g["STRIKE"].to_numpy(), call_bid=g["C_BID"].to_numpy(),
                          call_ask=g["C_ASK"].to_numpy(), put_bid=g["P_BID"].to_numpy(),
                          put_ask=g["P_ASK"].to_numpy(), T=T, quote_time=qt, expiry=settle,
                          underlying="SPX-EOD", spot=spot)
