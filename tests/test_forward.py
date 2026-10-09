import numpy as np
import pytest

from rnd.chain import InsufficientData, OptionChain, implied_forward
from rnd.synthetic import Lognormal, quote_chain


@pytest.mark.parametrize("seed", range(5))
def test_parity_forward_recovers_truth(seed):
    F, T, DF = 6700.0, 30 / 365, 0.995
    ch = quote_chain(Lognormal(F, 0.18, T), np.arange(4500, 8500, 25.0), T, DF=DF, seed=seed)
    f = implied_forward(ch)
    assert abs(f.F / F - 1) < 2e-4          # within ~1.3 index points
    assert abs(f.DF - DF) < 5e-3


def test_known_df_mode():
    F, T = 6700.0, 30 / 365
    ch = quote_chain(Lognormal(F, 0.18, T), np.arange(4500, 8500, 25.0), T, DF=0.99, seed=3)
    assert implied_forward(ch, DF=0.99).DF == 0.99


def test_insufficient_quotes_raise():
    ch = OptionChain(strikes=np.array([1.0, 2.0]), call_bid=np.array([1, np.nan]),
                     call_ask=np.array([1.1, np.nan]), put_bid=np.array([np.nan, 1]),
                     put_ask=np.array([np.nan, 1.1]), T=0.1)
    with pytest.raises(InsufficientData):
        implied_forward(ch)
