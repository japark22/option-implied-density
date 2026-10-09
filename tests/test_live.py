import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rnd.synthetic import Lognormal, quote_chain

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("live_fit", ROOT / "scripts" / "live_fit.py")
live = importlib.util.module_from_spec(spec)
spec.loader.exec_module(live)


def _payload(path):
    F, opts = 6700.0, []
    asof = "2026-10-08T16:14:59"
    for ymd, hours in (("261008", 0.0), ("261009", 23.75), ("261016", 191.75)):
        T = max(hours, 1.0) / (24 * 365)
        ch = quote_chain(Lognormal(F, 0.15, T), np.arange(6400, 7000, 10.0), T, rel_spread=0.02, seed=1)
        for i, k in enumerate(ch.strikes):
            for cp, b, a in (("C", ch.call_bid[i], ch.call_ask[i]), ("P", ch.put_bid[i], ch.put_ask[i])):
                opts.append({"option": f"SPXW{ymd}{cp}{int(k * 1000):08d}", "bid": float(b), "ask": float(a),
                             "last_trade_time": asof})
    path.write_text(json.dumps({"timestamp": "2026-10-09 02:00:00", "data": {"options": opts}}))


def test_expired_and_same_minute_expiries_are_skipped():
    df = pd.DataFrame({"root": ["SPXW"] * 3,
                       "expiry": [pd.Timestamp(d).date() for d in ("2026-10-08", "2026-10-09", "2026-10-16")]})
    asof = pd.Timestamp("2026-10-08 16:14:59", tz="America/New_York")
    picked = live.pick_expiries(df, asof)
    assert str(picked["front"]) == "2026-10-09" and str(picked["week"]) == "2026-10-16"


def test_run_writes_history_once_per_asof(tmp_path, monkeypatch):
    monkeypatch.setattr(live, "OUT", tmp_path / "live")
    f = tmp_path / "chain.json"
    _payload(f)
    for _ in range(2):                                   # second run must not duplicate rows
        monkeypatch.setattr(sys, "argv", ["live_fit.py", "--file", str(f), "--boot", "3"])
        live.main()
    h = pd.read_csv(tmp_path / "live" / "history.csv")
    assert len(h) == 2 and set(h["label"]) == {"front", "week"}
    assert (h["hard_arbitrage"] == 0).all() and (h["negative_mass"] < 1e-6).all()
    assert abs(h.loc[h["label"] == "front", "hours_to_expiry"].iloc[0] - 23.75) < 1e-6
    assert (tmp_path / "live" / "latest_front.png").exists()
