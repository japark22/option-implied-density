"""Implied-volatility smile fitters in total-variance space.

Two deliberately different fitters are provided so that *model risk* can be
measured, not just assumed away:

* ``SVISmile``  - Gatheral's raw SVI (5 parameters, parametric wings).
* ``SplineSmile`` - penalised cubic smoothing spline (non-parametric) with
  linear total-variance wings.

Both are fitted to out-of-the-money mid prices with residuals expressed in
units of the quoted half-spread, so "a good fit" has a market meaning: the
model price sits inside the bid-ask.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.interpolate import make_smoothing_spline
from scipy.optimize import least_squares
from scipy.stats import norm

from .black import black_call


class Smile:
    """Interface: total variance w(k) and its first two k-derivatives."""

    def w(self, k):  # pragma: no cover - interface
        raise NotImplementedError

    def dw(self, k):  # pragma: no cover
        raise NotImplementedError

    def d2w(self, k):  # pragma: no cover
        raise NotImplementedError

    def iv(self, k, T):
        return np.sqrt(np.maximum(self.w(k), 0.0) / T)

    def durrleman_g(self, k):
        """Butterfly-arbitrage function g(k); g < 0 means a negative density."""
        k = np.asarray(k, float)
        w, w1, w2 = self.w(k), self.dw(k), self.d2w(k)
        w = np.maximum(w, 1e-16)
        return (1 - k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2


# --------------------------------------------------------------------- SVI
@dataclass
class SVISmile(Smile):
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def w(self, k):
        x = np.asarray(k, float) - self.m
        return self.a + self.b * (self.rho * x + np.sqrt(x * x + self.sigma ** 2))

    def dw(self, k):
        x = np.asarray(k, float) - self.m
        return self.b * (self.rho + x / np.sqrt(x * x + self.sigma ** 2))

    def d2w(self, k):
        x = np.asarray(k, float) - self.m
        s2 = self.sigma ** 2
        return self.b * s2 / (x * x + s2) ** 1.5

    def params(self):
        return dict(a=self.a, b=self.b, rho=self.rho, m=self.m, sigma=self.sigma)


# ------------------------------------------------------------------ spline
@dataclass
class SplineSmile(Smile):
    """Smoothing spline inside the quoted range, C2-continuous wings outside it.

    In the outward coordinate d >= 0 (u(d) = w(k_edge +/- d)) the wing has
    curvature  u''(d) = (u2 + beta d) exp(-a d),  starting from the spline's
    edge value, slope and curvature (so w, w', w'' and hence the density are
    continuous) and converging to an asymptotic slope clipped to Lee's moment
    bound [0, 2]. ``beta`` is solved in closed form so the limit slope is hit
    exactly; beta = 0 when no clipping is needed.
    """
    spline: object
    k_lo: float
    k_hi: float

    def __post_init__(self):
        self._d1 = self.spline.derivative(1)
        self._d2 = self.spline.derivative(2)
        span = max(self.k_hi - self.k_lo, 1e-6)
        a = 4.0 / span
        self._edges = {}
        for side, ke, sign in (("lo", self.k_lo, -1.0), ("hi", self.k_hi, 1.0)):
            u0 = float(self.spline(ke))
            u1 = sign * float(self._d1(ke))
            u2 = float(self._d2(ke))
            target = float(np.clip(u1 + u2 / a, 0.0, 2.0))
            beta = a * a * (target - u1 - u2 / a)
            self._edges[side] = (ke, sign, a, u0, u1, u2, beta)

    def _wing(self, k, side, order):
        ke, sign, a, u0, u1, u2, beta = self._edges[side]
        d = np.maximum(sign * (np.asarray(k, float) - ke), 0.0)
        E = np.exp(-a * d)
        if order == 2:
            return (u2 + beta * d) * E
        du = u1 + u2 * (1 - E) / a + beta * (1 - E * (1 + a * d)) / a ** 2
        if order == 1:
            return sign * du
        return (u0 + u1 * d + u2 * (d / a - (1 - E) / a ** 2)
                + beta / a ** 2 * (d - (2 - 2 * E - a * d * E) / a))

    def _eval(self, k, order, inner_fn):
        k = np.asarray(k, float)
        inner = inner_fn(np.clip(k, self.k_lo, self.k_hi))
        out = np.where(k < self.k_lo, self._wing(k, "lo", order), inner)
        return np.where(k > self.k_hi, self._wing(k, "hi", order), out)

    def w(self, k):
        return np.maximum(self._eval(k, 0, self.spline), 1e-12)

    def dw(self, k):
        return self._eval(k, 1, self._d1)

    def d2w(self, k):
        return self._eval(k, 2, self._d2)


# ----------------------------------------------------------------- fitting
@dataclass
class SmileFit:
    smile: Smile
    method: str
    rmse_halfspreads: float      # RMS pricing error in units of half-spread
    frac_inside_spread: float    # share of fitted prices inside [bid, ask]
    n_quotes: int
    lam: float | None = None     # spline penalty actually used (None for SVI)


def _price_residual_scale(otm: pd.DataFrame) -> np.ndarray:
    return np.maximum(0.5 * (otm["ask"] - otm["bid"]).to_numpy(), 1e-8)


def _diagnostics(smile: Smile, otm: pd.DataFrame, F: float, method: str) -> SmileFit:
    K = otm["strike"].to_numpy()
    k = otm["k"].to_numpy()
    c = black_call(F, K, smile.w(k))
    model = np.where(otm["is_call"].to_numpy(), c, c - (F - K))
    half = _price_residual_scale(otm)
    r = (model - otm["mid"].to_numpy()) / half
    inside = (model >= otm["bid"].to_numpy() - 1e-12) & (model <= otm["ask"].to_numpy() + 1e-12)
    return SmileFit(smile=smile, method=method, rmse_halfspreads=float(np.sqrt(np.mean(r ** 2))),
                    frac_inside_spread=float(inside.mean()), n_quotes=int(len(otm)))


def fit_svi(otm: pd.DataFrame, F: float, n_starts: int = 6, seed: int = 0) -> SmileFit:
    """Fit raw SVI by minimising OTM price errors scaled by half-spread."""
    if len(otm) < 5:
        from .chain import InsufficientData
        raise InsufficientData("SVI needs at least 5 OTM quotes")
    K = otm["strike"].to_numpy()
    k = otm["k"].to_numpy()
    is_call = otm["is_call"].to_numpy()
    mid = otm["mid"].to_numpy()
    half = _price_residual_scale(otm)
    wmid = otm["w_mid"].to_numpy()

    span = max(k.max() - k.min(), 1e-4)
    w_atm = float(np.interp(0.0, k, wmid)) if k.min() < 0 < k.max() else float(np.median(wmid))
    w_max = float(np.nanmax(wmid)) * 4 + 1e-8

    lb = np.array([-w_max, 1e-10, -0.999, k.min() - span, 1e-5 * span])
    ub = np.array([w_max, 10 * w_max / span + 1e-6, 0.999, k.max() + span, 5 * span])

    def resid(p):
        s = SVISmile(*p)
        w = s.w(k)
        c = black_call(F, K, w)
        model = np.where(is_call, c, c - (F - K))
        r = (model - mid) / half
        # penalties: non-negative minimum variance, Lee wing bound
        minw = p[0] + p[1] * p[4] * np.sqrt(1 - p[2] ** 2)
        pen1 = 1e3 * max(0.0, -minw) / (w_atm + 1e-12)
        pen2 = 1e3 * max(0.0, p[1] * (1 + abs(p[2])) - 2.0)
        return np.concatenate([r, [pen1, pen2]])

    rng = np.random.default_rng(seed)
    starts = [np.array([0.5 * w_atm, 0.5 * w_atm / (0.2 * span), -0.5, 0.0, 0.2 * span])]
    for _ in range(n_starts - 1):
        starts.append(np.array([
            rng.uniform(0.0, 1.0) * w_atm,
            rng.uniform(0.1, 2.0) * w_atm / (0.2 * span),
            rng.uniform(-0.95, 0.5),
            rng.uniform(-0.3, 0.3) * span,
            rng.uniform(0.05, 1.0) * span,
        ]))
    best = None
    for x0 in starts:
        x0 = np.clip(x0, lb + 1e-12, ub - 1e-12)
        try:
            sol = least_squares(resid, x0, bounds=(lb, ub), x_scale="jac", max_nfev=4000)
        except Exception:  # noqa: BLE001 - try other starts
            continue
        if best is None or sol.cost < best.cost:
            best = sol
    if best is None:
        raise RuntimeError("SVI fit failed for all starting points")
    return _diagnostics(SVISmile(*best.x), otm, F, "svi")


def fit_spline(otm: pd.DataFrame, F: float, lam: float | None = None,
               target_halfspreads: float = 1 / np.sqrt(3)) -> SmileFit:
    """Penalised cubic smoothing spline of total variance.

    Weights are (vega / half-spread)^2, so the data term measures pricing error
    in half-spreads. The roughness penalty ``lam`` is chosen by the
    *discrepancy principle*: the smoothest curve whose RMS pricing error is
    within ``target_halfspreads`` half-spreads, i.e. no smoother than the
    bid-ask allows and no rougher than it requires. The default 1/sqrt(3) is
    the RMS distance of a fair value spread uniformly over [bid, ask] from the
    mid, measured in half-spreads. Pass ``lam`` to override.
    """
    if len(otm) < 5:
        from .chain import InsufficientData
        raise InsufficientData("spline needs at least 5 OTM quotes")
    K = otm["strike"].to_numpy()
    k = otm["k"].to_numpy()
    w = otm["w_mid"].to_numpy()
    sw = np.sqrt(np.maximum(w, 1e-16))
    d1 = (np.log(F / K) + 0.5 * w) / sw
    vega_w = F * norm.pdf(d1) / (2 * sw)           # dc/dw
    half = _price_residual_scale(otm)
    weights = (vega_w / half) ** 2
    weights = weights / weights.mean()

    def build(l):
        spl = make_smoothing_spline(k, w, w=weights, lam=l)
        smile = SplineSmile(spline=spl, k_lo=float(k.min()), k_hi=float(k.max()))
        fit = _diagnostics(smile, otm, F, "spline")
        fit.lam = float(l)
        return fit

    if lam is not None:
        return build(lam)
    # scale-aware lambda grid: penalty ~ lam * int w''^2, w'' ~ w_atm / span^2
    span = max(k.max() - k.min(), 1e-4)
    base = span ** 3          # makes lam dimensionless across maturities
    fits = []
    for l in base * np.logspace(-10, 2, 31):
        try:
            fits.append((l, build(l)))
        except Exception:  # noqa: BLE001
            continue
    if not fits:
        raise RuntimeError("spline fit failed for all penalties")
    ok = [(l, f) for l, f in fits if f.rmse_halfspreads <= target_halfspreads]
    if ok:
        return max(ok, key=lambda t: t[0])[1]
    return min(fits, key=lambda t: t[1].rmse_halfspreads)[1]


FITTERS = {"svi": fit_svi, "spline": fit_spline}
