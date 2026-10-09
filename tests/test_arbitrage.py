import numpy as np

from rnd.arbitrage import quote_arbitrage, summarize
from rnd.chain import implied_forward
from rnd.synthetic import Lognormal, quote_chain


def _chain():
    T = 30 / 365
    return quote_chain(Lognormal(6700.0, 0.18, T), np.arange(5500, 8000, 25.0), T, seed=0)


def test_clean_chain_has_no_hard_arbitrage():
    ch = _chain()
    assert summarize(quote_arbitrage(ch, implied_forward(ch)))["n_hard"] == 0


def test_injected_butterfly_is_detected():
    ch = _chain()
    fwd = implied_forward(ch)
    i = int(np.argmin(np.abs(ch.strikes - 7000)))
    cb, ca = ch.call_bid.copy(), ch.call_ask.copy()
    bump = 0.5 * (ca[i - 1] + ca[i + 1]) - cb[i] + 1.0      # body bid above wing asks
    cb[i] += bump
    ca[i] += bump
    bad = ch.with_quotes(cb, ca, ch.put_bid, ch.put_ask)
    v = quote_arbitrage(bad, fwd)
    hard = v[v["hard"] & (v["type"] == "call_butterfly")]
    assert any(abs(s[1] - ch.strikes[i]) < 1e-9 for s in hard["strikes"])
