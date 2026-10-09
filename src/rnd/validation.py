"""Out-of-sample evaluation of density and probability forecasts.

* PIT + Berkowitz (2001) LR test - is the realised outcome distributed as the
  forecast density said it would be? (Risk-neutral densities are *expected* to
  fail this by the risk premium; the size and sign of the failure is the
  interesting part.)
* Log score / Brier score for bucket and binary forecasts.
* Diebold-Mariano test with Newey-West variance for comparing two forecasters
  on the same outcomes (e.g. options-implied vs prediction market).
* Reliability table for calibration plots.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


def pit(cdf_values) -> np.ndarray:
    """PIT values u_t = F_t(x_t); pass the forecast CDF evaluated at the outcome."""
    return np.clip(np.asarray(cdf_values, float), 1e-10, 1 - 1e-10)


def berkowitz_test(u) -> dict:
    """Berkowitz (2001) LR test of z = Phi^{-1}(u) ~ iid N(0, 1).

    Alternative: z_t = mu + rho (z_{t-1} - mu) + sigma e_t. Returns the joint
    3-dof LR statistic and the 2-dof (mean/variance only) version.
    """
    z = stats.norm.ppf(pit(u))
    n = len(z)
    if n < 10:
        return dict(n=n, lr3=np.nan, p3=np.nan, lr2=np.nan, p2=np.nan, mu=np.nan, sigma=np.nan, rho=np.nan,
                    note="insufficient data (n < 10)")

    def nll_ar(p):
        mu, log_s, rho = p
        s = np.exp(log_s)
        e = (z[1:] - mu) - rho * (z[:-1] - mu)
        # exact first observation from the stationary distribution
        v0 = s ** 2 / max(1 - rho ** 2, 1e-6)
        ll = stats.norm.logpdf(z[0], mu, np.sqrt(v0)) + stats.norm.logpdf(e, 0, s).sum()
        return -ll

    res = minimize(nll_ar, x0=[z.mean(), np.log(z.std() + 1e-9), 0.0],
                   bounds=[(None, None), (None, None), (-0.99, 0.99)])
    ll1 = -res.fun
    ll0 = stats.norm.logpdf(z).sum()
    mu_i, s_i = z.mean(), z.std()
    ll1_iid = stats.norm.logpdf(z, mu_i, s_i).sum()
    lr3 = 2 * (ll1 - ll0)
    lr2 = 2 * (ll1_iid - ll0)
    return dict(n=n, lr3=lr3, p3=stats.chi2.sf(lr3, 3), lr2=lr2, p2=stats.chi2.sf(lr2, 2),
                mu=res.x[0], sigma=float(np.exp(res.x[1])), rho=res.x[2])


def ks_uniform(u) -> dict:
    r = stats.kstest(pit(u), "uniform")
    return dict(ks=r.statistic, p=r.pvalue, n=len(u))


def log_score(p_realized, floor=1e-6) -> np.ndarray:
    """Log score of the probability assigned to what actually happened (higher = better)."""
    return np.log(np.maximum(np.asarray(p_realized, float), floor))


def brier(p, y) -> np.ndarray:
    return (np.asarray(p, float) - np.asarray(y, float)) ** 2


def newey_west_var(x, lags=None) -> float:
    x = np.asarray(x, float) - np.mean(x)
    n = len(x)
    if lags is None:
        lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    v = np.dot(x, x) / n
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        v += 2 * w * np.dot(x[L:], x[:-L]) / n
    return v


def diebold_mariano(loss_a, loss_b, lags=None) -> dict:
    """DM test of equal expected loss. d = loss_a - loss_b; negative mean => A better."""
    d = np.asarray(loss_a, float) - np.asarray(loss_b, float)
    d = d[np.isfinite(d)]
    n = len(d)
    if n < 10:
        return dict(n=n, mean_diff=np.nan, dm=np.nan, p=np.nan, note="insufficient data (n < 10)")
    v = newey_west_var(d, lags)
    dm = d.mean() / np.sqrt(v / n)
    return dict(n=n, mean_diff=d.mean(), dm=dm, p=2 * stats.norm.sf(abs(dm)))


def reliability_table(p, y, bins=10) -> pd.DataFrame:
    p = np.asarray(p, float)
    y = np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        if m.sum() == 0:
            continue
        rows.append(dict(bin_lo=edges[b], bin_hi=edges[b + 1], n=int(m.sum()),
                         mean_forecast=p[m].mean(), freq=y[m].mean()))
    return pd.DataFrame(rows)
