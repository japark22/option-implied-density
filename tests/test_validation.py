import numpy as np
from scipy import stats

from rnd.validation import berkowitz_test, diebold_mariano, reliability_table


def test_berkowitz_accepts_calibrated_and_rejects_overconfident():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(500)
    good = berkowitz_test(stats.norm.cdf(x))
    bad = berkowitz_test(stats.norm.cdf(x, scale=0.6))     # forecast too narrow
    assert good["p3"] > 0.01
    assert bad["p3"] < 1e-6 and bad["sigma"] > 1.3


def test_berkowitz_small_sample_is_flagged():
    assert "insufficient" in berkowitz_test(np.full(5, 0.5))["note"]


def test_diebold_mariano_direction():
    rng = np.random.default_rng(1)
    a = rng.normal(0.0, 1, 400)
    b = a + 0.3 + rng.normal(0, 0.5, 400)               # B has higher loss
    r = diebold_mariano(a, b)
    assert r["mean_diff"] < 0 and r["p"] < 1e-3


def test_reliability_table_counts():
    t = reliability_table([0.05, 0.15, 0.95, 0.96], [0, 0, 1, 1], bins=10)
    assert t["n"].sum() == 4
