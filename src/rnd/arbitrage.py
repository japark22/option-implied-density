"""Static-arbitrage checks on *raw quotes* (before any smoothing).

A violation is only counted as a tradeable arbitrage if it survives the
bid-ask: e.g. a butterfly is an arbitrage only if buying the wings at the ask
and selling the body at the bid still produces a credit. Mid-price violations
are reported separately as "soft" (usually just noise).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .chain import ForwardEstimate, OptionChain


def _rows(kind, K, amount, mask):
    idx = np.where(mask)[0]
    return [dict(type=kind, strikes=tuple(float(x) for x in K[i]), amount=float(amount[i])) for i in idx]


def quote_arbitrage(chain: OptionChain, fwd: ForwardEstimate, tol: float = 1e-9) -> pd.DataFrame:
    """Return a table of hard (executable) and soft (mid) static-arbitrage violations."""
    out = []
    DF = fwd.DF
    K = chain.strikes
    for opt, bid, ask in (("call", chain.call_bid, chain.call_ask), ("put", chain.put_bid, chain.put_ask)):
        ok = np.isfinite(bid) & np.isfinite(ask) & (ask > 0)
        Kq, b, a = K[ok], np.nan_to_num(bid[ok]), ask[ok]
        m = 0.5 * (b + a)
        if len(Kq) < 3:
            continue
        # vertical (monotonicity): calls decrease in K, puts increase in K
        pairs = np.column_stack([Kq[:-1], Kq[1:]])
        if opt == "call":
            hard = b[1:] - a[:-1]          # buy low strike at ask, sell high strike at bid
            soft = m[1:] - m[:-1]
        else:
            hard = b[:-1] - a[1:]
            soft = m[:-1] - m[1:]
        out += _rows(f"{opt}_vertical", pairs, hard, hard > tol)
        out += _rows(f"{opt}_vertical_mid", pairs, soft, soft > tol)
        # slope bound: |dP/dK| <= DF
        dK = Kq[1:] - Kq[:-1]
        hard_s = (b[:-1] - a[1:] if opt == "call" else b[1:] - a[:-1]) - DF * dK
        out += _rows(f"{opt}_slope_bound", pairs, hard_s, hard_s > tol)
        # butterfly (convexity) on consecutive triples, unequal spacing allowed
        K1, K2, K3 = Kq[:-2], Kq[1:-1], Kq[2:]
        lam = (K3 - K2) / (K3 - K1)
        trip = np.column_stack([K1, K2, K3])
        hard_b = b[1:-1] - (lam * a[:-2] + (1 - lam) * a[2:])
        soft_b = m[1:-1] - (lam * m[:-2] + (1 - lam) * m[2:])
        out += _rows(f"{opt}_butterfly", trip, hard_b, hard_b > tol)
        out += _rows(f"{opt}_butterfly_mid", trip, soft_b, soft_b > tol)
    df = pd.DataFrame(out, columns=["type", "strikes", "amount"])
    df["hard"] = ~df["type"].str.endswith("_mid")
    return df


def summarize(viol: pd.DataFrame) -> dict:
    if viol.empty:
        return dict(n_hard=0, n_soft=0, max_hard=0.0)
    hard = viol[viol["hard"]]
    return dict(n_hard=int(len(hard)), n_soft=int((~viol["hard"]).sum()),
                max_hard=float(hard["amount"].max()) if len(hard) else 0.0)
