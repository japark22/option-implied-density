import numpy as np
import pytest

from rnd import fit_rnd
from rnd.synthetic import Heston, Lognormal, LognormalMixture, quote_chain

F = 6700.0


def _err(res, model):
    x = res.rnd.grid
    l1 = np.trapezoid(np.abs(res.rnd.pdf_clean - model.pdf(x)), x)
    ks = np.max(np.abs(res.rnd.cdf(x) - model.cdf(x)))
    return l1, ks


def test_analytic_density_matches_finite_difference():
    T = 30 / 365
    ch = quote_chain(Heston(F, T), np.arange(4500, 8000, 25.0), T, seed=0)
    r = fit_rnd(ch, "svi").rnd
    K = np.linspace(6000, 7300, 50)
    h = 0.5
    fd = (r.cdf(K + h) - r.cdf(K - h)) / (2 * h)
    assert np.allclose(fd, r.pdf(K), rtol=1e-3, atol=1e-8)


@pytest.mark.parametrize("method", ["svi", "spline"])
def test_lognormal_recovery(method):
    T = 30 / 365
    m = Lognormal(F, 0.18, T)
    r = fit_rnd(quote_chain(m, np.arange(4500, 8500, 25.0), T, DF=0.995, seed=1), method)
    l1, ks = _err(r, m)
    assert l1 < 0.01 and ks < 0.002
    d = r.rnd.diagnostics()
    assert d["negative_mass"] < 1e-6 and abs(d["martingale_error"]) < 1e-4


def test_heston_recovery():
    T = 30 / 365
    m = Heston(F, T)
    r = fit_rnd(quote_chain(m, np.arange(4500, 8000, 25.0), T, DF=0.995, seed=1), "auto")
    l1, ks = _err(r, m)
    assert l1 < 0.04 and ks < 0.01
    assert r.rnd.moments()["skew"] < -0.5      # equity-style negative skew is recovered


def test_svi_cannot_see_bimodality_but_auto_can():
    T = 7 / 365
    m = LognormalMixture(F, (0.5, 0.5), (0.97, 1.03), (0.10, 0.10), T)
    ch = quote_chain(m, np.arange(5800, 7600, 10.0), T, DF=0.998, seed=1)
    svi = fit_rnd(ch, "svi")
    auto = fit_rnd(ch, "auto")
    assert svi.fit.rmse_halfspreads > 5           # the diagnostic flags the bad model
    assert auto.fit.method == "spline"
    l1, _ = _err(auto, m)
    assert l1 < 0.08
    x = auto.rnd.grid
    p = auto.rnd.pdf_clean
    peaks = np.where((p[1:-1] > p[:-2]) & (p[1:-1] > p[2:]) & (p[1:-1] > 0.2 * p.max()))[0]
    assert len(peaks) == 2


def test_bucket_probabilities_sum_to_one():
    T = 5 / (24 * 365)
    m = LognormalMixture(F, (0.8, 0.2), (1, 1), (0.12, 0.30), T)
    r = fit_rnd(quote_chain(m, np.arange(6550, 6850, 5.0), T, seed=2), "auto").rnd
    edges = np.arange(6500, 6900, 25.0)
    probs = r.prob_between(edges[:-1], edges[1:])
    total = probs.sum() + r.prob_below(edges[0]) + r.prob_above(edges[-1])
    assert abs(total - 1) < 1e-9


def test_spline_wings_are_c2_continuous():
    T = 7 / 365
    m = LognormalMixture(F, (0.5, 0.5), (0.97, 1.03), (0.10, 0.10), T)
    r = fit_rnd(quote_chain(m, np.arange(5800, 7600, 10.0), T, DF=0.998, seed=1), "spline")
    sm = r.fit.smile
    for ke in (sm.k_lo, sm.k_hi):
        for f in (sm.w, sm.dw, sm.d2w):
            assert abs(f(ke - 1e-7) - f(ke + 1e-7)) < 1e-4 * (1 + abs(f(ke)))
    far = 50 * (sm.k_hi - sm.k_lo)
    assert -1e-9 <= sm.dw(sm.k_hi + far) <= 2 + 1e-9        # Lee bound, right wing
    assert -2 - 1e-9 <= sm.dw(sm.k_lo - far) <= 1e-9        # Lee bound, left wing
    # numerical derivative of w matches dw in the wings (closed forms are consistent)
    for k0 in (sm.k_hi + 0.03, sm.k_lo - 0.03):
        h = 1e-6
        assert abs((sm.w(k0 + h) - sm.w(k0 - h)) / (2 * h) - sm.dw(k0)) < 1e-6
        assert abs((sm.dw(k0 + h) - sm.dw(k0 - h)) / (2 * h) - sm.d2w(k0)) < 1e-4
