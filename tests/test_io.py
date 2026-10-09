import pandas as pd

from rnd.io.cboe import chain_from_table, parse_cboe_delayed, parse_symbol


def test_symbol_parsing_distinguishes_roots():
    a = parse_symbol("SPXW261009C07500000")
    b = parse_symbol("SPX261016P00200000")
    assert a["root"] == "SPXW" and a["strike"] == 7500.0 and a["cp"] == "C"
    assert b["root"] == "SPX" and b["strike"] == 200.0 and str(b["expiry"]) == "2026-10-16"


def test_payload_to_chain():
    # Fixture built from the documented field names (synthetic values).
    opts = []
    for k in (6600, 6650, 6700, 6750, 6800):
        for cp in "CP":
            opts.append({"option": f"SPXW261009{cp}{k * 1000:08d}", "bid": 10.0, "ask": 10.5,
                         "bid_size": 5, "ask_size": 5, "iv": 0.2, "open_interest": 1, "volume": 1,
                         "last_trade_price": 10.2, "last_trade_time": "2026-10-09T11:00:00"})
    payload = {"timestamp": "2026-10-09 11:15:00", "data": {"options": opts, "current_price": 6700.0}}
    df, meta = parse_cboe_delayed(payload)
    assert len(df) == 10 and meta["current_price"] == 6700.0
    qt = pd.Timestamp("2026-10-09 11:15", tz="America/New_York")
    ch = chain_from_table(df, "2026-10-09", qt)
    assert len(ch.strikes) == 5
    assert abs(ch.T * 365 * 24 - (16 - 11.25)) < 1e-6
