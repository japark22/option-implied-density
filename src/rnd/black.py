"""Black-76 pricing on the forward, and a vectorised implied-volatility solver.

Everything in this package works on *undiscounted* (forward) prices:
    c(K) = C(K) / DF = F N(d1) - K N(d2)
so the discount factor never enters the density maths and only appears when
converting quoted (discounted) premiums.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtr  # standard normal CDF, vectorised and fast

_SQRT_EPS = 1e-12


def black_call(F, K, w):
    """Undiscounted Black-76 call price.

    Parameters
    ----------
    F : forward
    K : strike(s)
    w : total implied variance sigma^2 * T (same shape as K or scalar)
    """
    F = np.asarray(F, dtype=float)
    K = np.asarray(K, dtype=float)
    w = np.maximum(np.asarray(w, dtype=float), _SQRT_EPS)
    sw = np.sqrt(w)
    d1 = (np.log(F / K) + 0.5 * w) / sw
    d2 = d1 - sw
    return F * ndtr(d1) - K * ndtr(d2)


def black_put(F, K, w):
    """Undiscounted Black-76 put price (via put-call parity)."""
    return black_call(F, K, w) - (np.asarray(F, float) - np.asarray(K, float))


def implied_total_variance(price, F, K, is_call, tol=1e-12, max_iter=200):
    """Invert Black-76 for total variance w = sigma^2 T.

    Vectorised bisection on sigma*sqrt(T) (robust, monotone). Prices outside
    the no-arbitrage bounds return NaN rather than a fake volatility.

    Parameters
    ----------
    price : undiscounted option price(s)
    F : forward
    K : strike(s)
    is_call : bool array (True = call, False = put)
    """
    price = np.atleast_1d(np.asarray(price, dtype=float))
    K = np.broadcast_to(np.asarray(K, dtype=float), price.shape).copy()
    is_call = np.broadcast_to(np.asarray(is_call, dtype=bool), price.shape)
    F = float(F)

    # Work with the call-equivalent price via parity.
    c = np.where(is_call, price, price + (F - K))
    lower = np.maximum(F - K, 0.0)
    upper = np.full_like(c, F)
    ok = np.isfinite(c) & (c > lower + 1e-14) & (c < upper - 1e-14)

    lo = np.full_like(c, 1e-6)   # sigma*sqrt(T) bounds
    hi = np.full_like(c, 5.0)
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        val = black_call(F, K, mid ** 2)
        too_high = val > c
        hi = np.where(too_high, mid, hi)
        lo = np.where(too_high, lo, mid)
        if np.nanmax(np.where(ok, hi - lo, 0.0)) < tol:
            break
    s = 0.5 * (lo + hi)
    out = s ** 2
    out[~ok] = np.nan
    return out
