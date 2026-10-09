"""Models with a *known* terminal density, used for known-answer tests.

Each model exposes ``pdf(x)``, ``cdf(x)`` and undiscounted ``call(K)`` that are
mutually consistent, so a recovered density can be scored against the truth.

* ``Lognormal``         - Black-Scholes world (the trivial case).
* ``LognormalMixture``  - fat tails / bimodal "event" distributions (FOMC, CPI).
* ``Heston``            - stochastic volatility with skew; density by Fourier
                          inversion, call prices by integrating the payoff
                          against that density on a fine grid.

``quote_chain`` turns any model into a realistic bid/ask chain (SPX-style tick
rules, spreads widening with price, random placement of fair value inside the
spread, no-bid strikes in the far wings).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .black import black_call
from .chain import OptionChain


@dataclass
class Lognormal:
    F: float
    sigma: float
    T: float

    @property
    def w(self):
        return self.sigma ** 2 * self.T

    def pdf(self, x):
        x = np.asarray(x, float)
        s = np.sqrt(self.w)
        mu = np.log(self.F) - 0.5 * self.w
        return norm.pdf((np.log(x) - mu) / s) / (x * s)

    def cdf(self, x):
        s = np.sqrt(self.w)
        mu = np.log(self.F) - 0.5 * self.w
        return norm.cdf((np.log(np.asarray(x, float)) - mu) / s)

    def call(self, K):
        return black_call(self.F, K, self.w)


@dataclass
class LognormalMixture:
    """sum_i p_i * Lognormal(F_i, sigma_i). Forward = sum p_i F_i (rescaled to F)."""
    F: float
    weights: tuple
    rel_fwds: tuple      # component forwards relative to F (rescaled to be a martingale)
    sigmas: tuple
    T: float

    def __post_init__(self):
        p = np.asarray(self.weights, float)
        p = p / p.sum()
        r = np.asarray(self.rel_fwds, float)
        r = r / np.dot(p, r)
        self._p, self._F = p, self.F * r
        self._c = [Lognormal(Fi, si, self.T) for Fi, si in zip(self._F, self.sigmas)]

    def pdf(self, x):
        return sum(p * c.pdf(x) for p, c in zip(self._p, self._c))

    def cdf(self, x):
        return sum(p * c.cdf(x) for p, c in zip(self._p, self._c))

    def call(self, K):
        return sum(p * c.call(K) for p, c in zip(self._p, self._c))


@dataclass
class Heston:
    F: float
    T: float
    v0: float = 0.04
    kappa: float = 1.5
    theta: float = 0.04
    xi: float = 0.6
    rho: float = -0.7
    n_x: int = 4001
    u_max_sd: float = 14.0   # integrate the CF up to u = u_max_sd / sd
    n_u: int = 4001

    def __post_init__(self):
        sd = np.sqrt(max(self.v0, self.theta) * self.T)
        y = np.linspace(-12 * sd, 8 * sd, self.n_x)          # y = ln(S/F)
        u = np.linspace(1e-8, self.u_max_sd / sd, self.n_u)
        phi = self._cf(u)
        fy = np.empty_like(y)
        for i in range(0, len(y), 400):                       # chunk to bound memory
            blk = y[i:i + 400]
            integrand = np.real(np.exp(-1j * np.outer(blk, u)) * phi[None, :])
            fy[i:i + 400] = np.trapezoid(integrand, u, axis=1) / np.pi
        fy = np.maximum(fy, 0.0)
        fy /= np.trapezoid(fy, y)
        self._x = y + np.log(self.F)
        fx = fy
        self._S = np.exp(self._x)
        self._q = fx / self._S                     # density in S
        # cumulative helpers for prices: c(K) = E[S 1{S>K}] - K P(S>K)
        dS = np.diff(self._S)
        mids_q = 0.5 * (self._q[1:] + self._q[:-1])
        mids_Sq = 0.5 * (self._S[1:] * self._q[1:] + self._S[:-1] * self._q[:-1])
        tail_p = np.concatenate([np.cumsum((mids_q * dS)[::-1])[::-1], [0.0]])
        tail_m = np.concatenate([np.cumsum((mids_Sq * dS)[::-1])[::-1], [0.0]])
        scale = self.F / tail_m[0]                 # enforce the martingale exactly
        self._tail_p = tail_p / tail_p[0]
        self._tail_m = tail_m * scale
        self._q_norm = self._q / tail_p[0]

    def _cf(self, u):
        """CF of ln S_T under the forward measure ("little Heston trap" form)."""
        k, th, xi, rho, v0, T = self.kappa, self.theta, self.xi, self.rho, self.v0, self.T
        iu = 1j * u
        b = k - rho * xi * iu
        d = np.sqrt(b ** 2 + xi ** 2 * (iu + u ** 2))
        g = (b - d) / (b + d)
        e = np.exp(-d * T)
        C = k * th / xi ** 2 * ((b - d) * T - 2 * np.log((1 - g * e) / (1 - g)))
        D = (b - d) / xi ** 2 * (1 - e) / (1 - g * e)
        return np.exp(C + D * v0)       # CF of ln(S_T / F)

    def pdf(self, x):
        return np.interp(x, self._S, self._q_norm, left=0.0, right=0.0)

    def cdf(self, x):
        return 1.0 - np.interp(x, self._S, self._tail_p, left=1.0, right=0.0)

    def call(self, K):
        K = np.asarray(K, float)
        return np.interp(K, self._S, self._tail_m) - K * np.interp(K, self._S, self._tail_p)


def spx_tick(price):
    """SPX tick: 0.05 below $3, 0.10 at/above."""
    return np.where(price < 3.0, 0.05, 0.10)


def quote_chain(model, strikes, T, DF=1.0, rel_spread=0.01, min_spread=0.10,
                max_spread=None, seed=0, quote_time=None, expiry=None) -> OptionChain:
    """Turn a model into a quoted (discounted) chain with realistic microstructure.

    Fair value is placed uniformly inside the spread (not at mid). Options worth
    less than one tick are quoted 0 bid / 1 tick ask, as in the real wings.
    """
    rng = np.random.default_rng(seed)
    K = np.asarray(strikes, float)
    c = model.call(K)
    p = c - (model.F - K)
    out = {}
    for name, und in (("call", c), ("put", p)):
        fair = DF * np.maximum(und, 0.0)
        tick = spx_tick(fair)
        spread = np.maximum(min_spread, rel_spread * fair)
        if max_spread is not None:
            spread = np.minimum(spread, max_spread)
        spread = np.ceil(spread / tick) * tick
        off = rng.uniform(-0.5, 0.5, size=len(K)) * spread      # fair - mid
        bid = np.round((fair - 0.5 * spread - off) / tick) * tick   # symmetric rounding
        bid = np.where(bid > fair, bid - tick, bid)                  # keep fair inside
        bid = np.maximum(bid, 0.0)
        ask = bid + spread
        ask = np.where(ask < fair, ask + tick, ask)
        ask = np.maximum(ask, bid + tick)
        out[name] = (bid, ask)
    return OptionChain(strikes=K, call_bid=out["call"][0], call_ask=out["call"][1],
                       put_bid=out["put"][0], put_ask=out["put"][1], T=T,
                       quote_time=quote_time, expiry=expiry, underlying="SYNTH")
