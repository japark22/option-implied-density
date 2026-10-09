import numpy as np

from rnd.black import black_call, black_put, implied_total_variance


def test_put_call_parity():
    F, K, w = 100.0, np.array([80, 100, 120.0]), 0.04
    assert np.allclose(black_call(F, K, w) - black_put(F, K, w), F - K)


def test_iv_round_trip_calls_and_puts():
    F = 6700.0
    K = np.linspace(5000, 8000, 61)
    w = 0.03 + 0.1 * np.log(K / F) ** 2
    c = black_call(F, K, w)
    p = black_put(F, K, w)
    assert np.allclose(implied_total_variance(c, F, K, True), w, rtol=1e-6)
    assert np.allclose(implied_total_variance(p, F, K, False), w, rtol=1e-6)


def test_iv_rejects_arbitrage_prices():
    w = implied_total_variance(np.array([5.0, 200.0]), 100.0, np.array([90.0, 90.0]), True)
    assert np.isnan(w).all()   # below intrinsic, above the forward
