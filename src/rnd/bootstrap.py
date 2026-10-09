"""Quote-uncertainty bootstrap: error bars for anything computed from the RND.

A quote is an interval, not a number. Each replicate redraws every two-sided
quote's "true" price uniformly inside [bid, ask], then reruns the *whole*
pipeline (parity forward -> OTM table -> smile fit -> density). The spread of
any functional across replicates (a bucket probability, a quantile, the
density itself) is the uncertainty attributable to the bid-ask alone.

Note what this does *not* capture: smile-model misspecification. That is
measured separately by comparing fitters (see ``model_spread``), and the
known-answer study in ``scripts/known_answer_study.py`` quantifies how much
coverage is lost when the model is wrong.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, List

import numpy as np

from .chain import OptionChain
from .density import FitResult, fit_rnd


@dataclass
class RNDEnsemble:
    base: FitResult
    reps: List[FitResult] = field(repr=False)
    n_failed: int = 0

    def functional(self, fn: Callable) -> np.ndarray:
        """Evaluate ``fn(rnd)`` on every replicate -> array (n_reps, ...)."""
        return np.array([fn(r.rnd) for r in self.reps])

    def interval(self, fn: Callable, q=(5, 95)):
        """(point estimate on unperturbed mids, lower, upper) percentiles."""
        vals = self.functional(fn)
        lo, hi = np.percentile(vals, q, axis=0)
        return fn(self.base.rnd), lo, hi

    def prob_between(self, a, b, q=(5, 95)):
        return self.interval(lambda r: r.prob_between(a, b), q)

    def cdf_band(self, x, q=(5, 95)):
        return self.interval(lambda r: r.cdf(x), q)

    def pdf_band(self, x, q=(5, 95)):
        return self.interval(lambda r: np.maximum(r.pdf(x), 0.0), q)


def perturb_chain(chain: OptionChain, rng: np.random.Generator) -> OptionChain:
    """Collapse each two-sided quote to a point drawn uniformly in [bid, ask].

    Bid and ask are both set to that point +/- a tiny epsilon so downstream
    code (which needs bid < ask) treats it as a priced quote. One-sided
    (zero-bid) quotes are left untouched so the wing-truncation rule is the
    same in every replicate.
    """
    def draw(bid, ask):
        ok = np.isfinite(bid) & np.isfinite(ask) & (bid > 0) & (ask > bid)
        u = rng.uniform(size=len(bid))
        p = np.where(ok, bid + u * (ask - bid), np.nan)
        half = np.where(ok, 0.5 * (ask - bid), 0.0)
        # keep the original half-spread as the residual scale for the fitter
        new_bid = np.where(ok, p - half, bid)
        new_ask = np.where(ok, p + half, ask)
        new_bid = np.where(ok & (new_bid <= 0), p * 0.5, new_bid)
        return new_bid, new_ask

    cb, ca = draw(chain.call_bid, chain.call_ask)
    pb, pa = draw(chain.put_bid, chain.put_ask)
    return chain.with_quotes(cb, ca, pb, pa)


def bootstrap_rnd(chain: OptionChain, n_boot: int = 200, method: str = "svi",
                  seed: int = 0, grid_n: int = 801, **fit_kw) -> RNDEnsemble:
    """Bootstrap the full pipeline over quote uncertainty.

    Model *selection* and *tuning* are done once on the observed quotes and
    then held fixed (``auto`` resolves to one fitter; the spline penalty is
    frozen). Re-tuning inside every replicate lets the penalty jump between
    grid values and inflates the bands with tuning noise rather than quote
    noise.
    """
    base = fit_rnd(chain, method=method, grid_n=grid_n, **fit_kw)
    chosen = base.fit.method
    kw = dict(fit_kw)
    if chosen == "spline" and "lam" not in kw:
        kw["lam"] = base.fit.lam
    rng = np.random.default_rng(seed)
    reps, failed = [], 0
    for _ in range(n_boot):
        try:
            reps.append(fit_rnd(perturb_chain(chain, rng), method=chosen, grid_n=grid_n, **kw))
        except Exception:  # noqa: BLE001 - a failed replicate is counted, not hidden
            failed += 1
    if not reps:
        raise RuntimeError("all bootstrap replicates failed")
    return RNDEnsemble(base=base, reps=reps, n_failed=failed)


def model_spread(chain: OptionChain, fn: Callable, methods=("svi", "spline")) -> dict:
    """Evaluate a functional under each smile model: a direct model-risk measure."""
    out = {}
    for m in methods:
        try:
            out[m] = fn(fit_rnd(chain, method=m).rnd)
        except Exception as e:  # noqa: BLE001
            out[m] = np.nan
            out[f"{m}_error"] = str(e)
    return out


@dataclass
class TotalUncertainty:
    """Quote uncertainty (bootstrap) combined with model uncertainty (two fitters).

    The interval for a functional is the union of the per-fitter bootstrap
    intervals; the point estimate comes from the fitter that prices the quotes
    best. The known-answer study shows why the union is needed: under a
    misspecified smile model the single-fitter bands under-cover badly.
    """
    ensembles: dict
    best: str

    def interval(self, fn: Callable, q=(5, 95)):
        pts, los, his = {}, [], []
        for m, ens in self.ensembles.items():
            p, lo, hi = ens.interval(fn, q)
            pts[m] = p
            los.append(lo)
            his.append(hi)
        return pts[self.best], np.minimum.reduce(los), np.maximum.reduce(his)

    def prob_between(self, a, b, q=(5, 95)):
        return self.interval(lambda r: r.prob_between(a, b), q)

    def cdf_band(self, x, q=(5, 95)):
        return self.interval(lambda r: r.cdf(x), q)


def total_uncertainty(chain: OptionChain, n_boot: int = 100, seed: int = 0, grid_n: int = 801,
                      methods=("svi", "spline"), max_rmse_halfspreads: float = 2.0) -> TotalUncertainty:
    ens = {}
    for i, m in enumerate(methods):
        try:
            ens[m] = bootstrap_rnd(chain, n_boot=n_boot, method=m, seed=seed + i, grid_n=grid_n)
        except Exception:  # noqa: BLE001 - a fitter may legitimately fail on a thin chain
            continue
    if not ens:
        raise RuntimeError("no smile fitter succeeded")
    best = min(ens, key=lambda m: ens[m].base.fit.rmse_halfspreads)
    # A fitter that cannot price the quotes is rejected, not averaged in.
    cut = max(max_rmse_halfspreads, 1.5 * ens[best].base.fit.rmse_halfspreads)
    ens = {m: e for m, e in ens.items() if e.base.fit.rmse_halfspreads <= cut}
    return TotalUncertainty(ensembles=ens, best=best)
