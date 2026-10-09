"""Option chain container, cleaning, parity-implied forward, OTM quote table."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Optional

import numpy as np
import pandas as pd

from .black import implied_total_variance


class InsufficientData(ValueError):
    """Raised when a chain does not carry enough clean quotes to fit."""


@dataclass
class OptionChain:
    """One expiry of quoted (discounted) option premiums.

    Arrays are aligned on ``strikes`` (sorted, unique). Missing quotes are NaN.
    ``T`` is time to expiry in years (only used to express total variance as an
    annualised IV for display; the density itself depends on total variance).
    """

    strikes: np.ndarray
    call_bid: np.ndarray
    call_ask: np.ndarray
    put_bid: np.ndarray
    put_ask: np.ndarray
    T: float
    quote_time: Optional[pd.Timestamp] = None
    expiry: Optional[pd.Timestamp] = None
    underlying: str = ""
    spot: Optional[float] = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        order = np.argsort(self.strikes)
        for name in ("strikes", "call_bid", "call_ask", "put_bid", "put_ask"):
            setattr(self, name, np.asarray(getattr(self, name), dtype=float)[order])
        if len(np.unique(self.strikes)) != len(self.strikes):
            raise ValueError("strikes must be unique")
        if not (self.T > 0):
            raise ValueError("T must be positive")

    # ------------------------------------------------------------------ build
    @classmethod
    def from_frame(cls, df: pd.DataFrame, T: float, **kw) -> "OptionChain":
        """Build from long format: columns strike, cp ('C'/'P'), bid, ask."""
        need = {"strike", "cp", "bid", "ask"}
        missing = need - set(df.columns)
        if missing:
            raise ValueError(f"missing columns: {missing}")
        d = df.copy()
        d["cp"] = d["cp"].str.upper().str[0]
        wide = d.pivot_table(index="strike", columns="cp", values=["bid", "ask"], aggfunc="last")
        strikes = wide.index.to_numpy(float)

        def col(side, cp):
            key = (side, cp)
            return wide[key].to_numpy(float) if key in wide.columns else np.full(len(strikes), np.nan)

        return cls(strikes=strikes, call_bid=col("bid", "C"), call_ask=col("ask", "C"),
                   put_bid=col("bid", "P"), put_ask=col("ask", "P"), T=T, **kw)

    def with_quotes(self, call_bid, call_ask, put_bid, put_ask) -> "OptionChain":
        return replace(self, call_bid=call_bid, call_ask=call_ask, put_bid=put_bid, put_ask=put_ask)

    # ------------------------------------------------------------- utilities
    @staticmethod
    def _valid(bid, ask):
        return np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)

    def mids(self):
        cv = self._valid(self.call_bid, self.call_ask)
        pv = self._valid(self.put_bid, self.put_ask)
        cm = np.where(cv, 0.5 * (self.call_bid + self.call_ask), np.nan)
        pm = np.where(pv, 0.5 * (self.put_bid + self.put_ask), np.nan)
        return cm, pm


@dataclass
class ForwardEstimate:
    F: float
    DF: float
    n_strikes: int
    resid_std: float


def implied_forward(chain: OptionChain, n_near: int = 20, DF: Optional[float] = None,
                    fix_df_below_years: float = 3 / 365) -> ForwardEstimate:
    """Forward and discount factor from put-call parity.

    C(K) - P(K) = DF * (F - K). We regress (C_mid - P_mid) on K over the
    ``n_near`` strikes closest to at-the-money (smallest |C-P|), weighting by
    the inverse combined spread, and drop >3 MAD outliers once. This avoids
    assuming a rate or dividend yield. If ``DF`` is given, only F is estimated.

    For expiries under ``fix_df_below_years`` (3 days) DF is fixed at 1: the
    true value differs from 1 by < 0.05% at any plausible rate, far less than
    the noise of estimating it from a handful of tick-rounded quotes.
    """
    if DF is None and chain.T < fix_df_below_years:
        DF = 1.0
    cm, pm = chain.mids()
    both = np.isfinite(cm) & np.isfinite(pm)
    if both.sum() < 3:
        raise InsufficientData("fewer than 3 strikes with two-sided call and put quotes")
    K = chain.strikes[both]
    y = (cm - pm)[both]
    spread = ((chain.call_ask - chain.call_bid) + (chain.put_ask - chain.put_bid))[both]
    near = np.argsort(np.abs(y))[: max(n_near, 3)]
    K, y, spread = K[near], y[near], spread[near]
    w = 1.0 / np.maximum(spread, 1e-6)

    def fit(K, y, w):
        if DF is None:
            A = np.column_stack([np.ones_like(K), K]) * np.sqrt(w)[:, None]
            coef, *_ = np.linalg.lstsq(A, y * np.sqrt(w), rcond=None)
            df_ = -coef[1]
            F_ = coef[0] / df_
        else:
            df_ = DF
            F_ = np.average(y / df_ + K, weights=w)
        return F_, df_

    F_, df_ = fit(K, y, w)
    resid = y - df_ * (F_ - K)
    mad = np.median(np.abs(resid - np.median(resid))) + 1e-12
    keep = np.abs(resid) <= 3 * 1.4826 * mad
    if keep.sum() >= 3 and keep.sum() < len(K):
        K, y, w = K[keep], y[keep], w[keep]
        F_, df_ = fit(K, y, w)
        resid = y - df_ * (F_ - K)
    if not (0.5 < df_ <= 1.05) or not np.isfinite(F_) or F_ <= 0:
        raise InsufficientData(f"implausible parity fit: F={F_:.4f}, DF={df_:.5f}")
    return ForwardEstimate(F=float(F_), DF=float(df_), n_strikes=int(len(K)),
                           resid_std=float(np.std(resid)))


def otm_table(chain: OptionChain, fwd: ForwardEstimate, zero_bid_stop: int = 2) -> pd.DataFrame:
    """Out-of-the-money quotes converted to undiscounted prices and total variance.

    Puts are used below the forward and calls at/above it. Moving away from the
    forward, strikes are dropped after ``zero_bid_stop`` consecutive zero-bid
    quotes (the same truncation rule the Cboe VIX methodology uses).
    """
    F, DF = fwd.F, fwd.DF
    K = chain.strikes
    is_call = K >= F
    bid = np.where(is_call, chain.call_bid, chain.put_bid) / DF
    ask = np.where(is_call, chain.call_ask, chain.put_ask) / DF

    keep = np.zeros(len(K), dtype=bool)
    for side in (np.where(~is_call)[0][::-1], np.where(is_call)[0]):  # walk outward
        zeros = 0
        for i in side:
            if not np.isfinite(bid[i]) or not np.isfinite(ask[i]) or ask[i] <= 0:
                continue
            if bid[i] <= 0:
                zeros += 1
                if zeros >= zero_bid_stop:
                    break
                continue
            zeros = 0
            if ask[i] > bid[i]:
                keep[i] = True

    d = pd.DataFrame({
        "strike": K[keep], "is_call": is_call[keep],
        "bid": bid[keep], "ask": ask[keep],
    })
    d["mid"] = 0.5 * (d["bid"] + d["ask"])
    d["k"] = np.log(d["strike"] / F)
    for col in ("bid", "ask", "mid"):
        d[f"w_{col}"] = implied_total_variance(d[col].to_numpy(), F, d["strike"].to_numpy(), d["is_call"].to_numpy())
    # bid can sit below intrinsic-free bound (e.g. 0.05 bid on a 0.03 fair) -> NaN w_bid; keep row if mid ok
    d = d[np.isfinite(d["w_mid"])].reset_index(drop=True)
    return d
