"""Breeden-Litzenberger risk-neutral density from a fitted smile.

Breeden & Litzenberger (1978): with undiscounted call prices c(K),
    CDF(K) = 1 + dc/dK,      q(K) = d^2 c / dK^2.
Differentiating noisy quotes directly is unstable, so we differentiate the
*fitted smile* analytically:

    dc/dK = -N(d2) + F phi(d1) / (2 sqrt(w)) * w'(k) / K
    q(K)  = g(k) * phi(d2) / (K sqrt(w))           (Gatheral's g function)

with k = ln(K/F), w = w(k) total variance, d2 = -k/sqrt(w) - sqrt(w)/2.
No finite differences, so no step-size tuning; a negative g(k) is reported as
a butterfly-arbitrage diagnostic rather than silently smoothed away.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd
from scipy.stats import norm

from .chain import ForwardEstimate, OptionChain, implied_forward, otm_table
from .smile import FITTERS, Smile, SmileFit


@dataclass
class RND:
    smile: Smile
    F: float
    DF: float
    T: float
    grid: np.ndarray = field(repr=False)
    pdf_raw: np.ndarray = field(repr=False)

    # ------------------------------------------------------------ analytic
    def _parts(self, K):
        K = np.asarray(K, float)
        k = np.log(K / self.F)
        w = np.maximum(self.smile.w(k), 1e-16)
        sw = np.sqrt(w)
        d1 = -k / sw + sw / 2
        d2 = d1 - sw
        return K, k, w, sw, d1, d2

    def cdf(self, K):
        K, k, w, sw, d1, d2 = self._parts(K)
        dc_dK = -norm.cdf(d2) + self.F * norm.pdf(d1) / (2 * sw) * self.smile.dw(k) / K
        return np.clip(1.0 + dc_dK, 0.0, 1.0)

    def pdf(self, K):
        K, k, w, sw, d1, d2 = self._parts(K)
        return self.smile.durrleman_g(k) * norm.pdf(d2) / (K * sw)

    # ------------------------------------------------------- probabilities
    def prob_below(self, x):
        return self.cdf(x)

    def prob_above(self, x):
        return 1.0 - self.cdf(x)

    def prob_between(self, a, b):
        return np.maximum(self.cdf(b) - self.cdf(a), 0.0)

    # ---------------------------------------------------- grid quantities
    @property
    def pdf_clean(self):
        p = np.maximum(self.pdf_raw, 0.0)
        mass = np.trapezoid(p, self.grid)
        return p / mass if mass > 0 else p

    def quantile(self, p):
        c = np.maximum.accumulate(self.cdf(self.grid))
        return np.interp(p, c, self.grid)

    def moments(self):
        x, p = self.grid, self.pdf_clean
        m = np.trapezoid(x * p, x)
        v = np.trapezoid((x - m) ** 2 * p, x)
        sd = np.sqrt(v)
        sk = np.trapezoid(((x - m) / sd) ** 3 * p, x)
        ku = np.trapezoid(((x - m) / sd) ** 4 * p, x)
        return dict(mean=m, std=sd, skew=sk, kurt=ku)

    def diagnostics(self):
        neg = np.trapezoid(np.maximum(-self.pdf_raw, 0.0), self.grid)
        g = self.smile.durrleman_g(np.log(self.grid / self.F))
        m = np.trapezoid(self.grid * self.pdf_clean, self.grid)
        return dict(
            negative_mass=float(neg),
            min_g=float(np.min(g)),
            tail_mass_lo=float(self.cdf(self.grid[0])),
            tail_mass_hi=float(1 - self.cdf(self.grid[-1])),
            martingale_error=float(m / self.F - 1.0),
        )


def make_rnd(smile: Smile, F: float, DF: float, T: float, n: int = 2001,
             k_range: Optional[tuple] = None, n_sd: float = 8.0) -> RND:
    """Evaluate the density on a strike grid spanning +/- ``n_sd`` ATM std devs."""
    if k_range is None:
        s = float(np.sqrt(max(smile.w(0.0), 1e-12)))
        k_range = (-n_sd * s, n_sd * s)
    grid = F * np.exp(np.linspace(k_range[0], k_range[1], n))
    grid = np.linspace(grid[0], grid[-1], n)     # uniform in K for integration
    tmp = RND(smile, F, DF, T, grid, np.zeros(n))
    tmp.pdf_raw = tmp.pdf(grid)
    return tmp


@dataclass
class FitResult:
    rnd: RND
    fit: SmileFit
    fwd: ForwardEstimate
    otm: pd.DataFrame

    def summary(self) -> dict:
        d = dict(method=self.fit.method, F=self.fwd.F, DF=self.fwd.DF,
                 n_quotes=self.fit.n_quotes, rmse_halfspreads=self.fit.rmse_halfspreads,
                 frac_inside_spread=self.fit.frac_inside_spread,
                 atm_iv=float(self.rnd.smile.iv(0.0, self.rnd.T)))
        d.update(self.rnd.diagnostics())
        d.update({f"m_{k}": v for k, v in self.rnd.moments().items()})
        return d


def fit_rnd(chain: OptionChain, method: str = "svi", fwd: Optional[ForwardEstimate] = None,
            grid_n: int = 2001, **fit_kw) -> FitResult:
    """Full pipeline: parity forward -> OTM quotes -> smile fit -> density.

    ``method`` is ``"svi"``, ``"spline"`` or ``"auto"``. ``auto`` fits both
    and keeps the one that prices the quotes better (lower RMS error in
    half-spreads); parametric SVI cannot represent multi-modal event
    distributions and is visibly rejected by this rule when that happens.
    """
    if method == "auto":
        fits = []
        for m in ("svi", "spline"):
            try:
                fits.append(fit_rnd(chain, m, fwd=fwd, grid_n=grid_n))
            except Exception:  # noqa: BLE001
                continue
        if not fits:
            raise RuntimeError("both smile fitters failed")
        return min(fits, key=lambda r: r.fit.rmse_halfspreads)
    if fwd is None:
        fwd = implied_forward(chain)
    otm = otm_table(chain, fwd)
    fit = FITTERS[method](otm, fwd.F, **fit_kw)
    k = otm["k"].to_numpy()
    s = float(np.sqrt(max(fit.smile.w(0.0), 1e-12)))
    k_range = (min(k.min(), -8 * s) - 2 * s, max(k.max(), 8 * s) + 2 * s)
    rnd = make_rnd(fit.smile, fwd.F, fwd.DF, chain.T, n=grid_n, k_range=k_range)
    return FitResult(rnd=rnd, fit=fit, fwd=fwd, otm=otm)
