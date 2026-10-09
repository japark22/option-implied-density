import numpy as np

from rnd.bootstrap import bootstrap_rnd, perturb_chain
from rnd.synthetic import Lognormal, quote_chain


def test_perturbed_quotes_stay_inside_original_spread():
    T = 30 / 365
    ch = quote_chain(Lognormal(6700.0, 0.18, T), np.arange(5500, 8000, 25.0), T, seed=0)
    p = perturb_chain(ch, np.random.default_rng(0))
    m_new = 0.5 * (p.call_bid + p.call_ask)
    ok = (ch.call_bid > 0) & (ch.call_ask > ch.call_bid)
    assert np.all(m_new[ok] >= ch.call_bid[ok] - 1e-12) and np.all(m_new[ok] <= ch.call_ask[ok] + 1e-12)


def test_bands_are_ordered_and_nonempty():
    T = 30 / 365
    m = Lognormal(6700.0, 0.18, T)
    ch = quote_chain(m, np.arange(5500, 8000, 25.0), T, seed=0)
    ens = bootstrap_rnd(ch, n_boot=30, method="svi", seed=1)
    pt, lo, hi = ens.prob_between(6600.0, 6800.0)
    assert lo <= pt <= hi and hi - lo > 0
    assert ens.n_failed == 0
