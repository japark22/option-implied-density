"""Fit today's SPXW density from Cboe's public delayed quotes and plot it.

    python scripts/fit_cboe_snapshot.py                 # nearest PM-settled expiry
    python scripts/fit_cboe_snapshot.py --expiry 2026-10-16 --save chain.json

Run during US market hours; outside them the chain is the previous close.
Writes results/figures/spxw_<expiry>.png and prints fit diagnostics.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnd.arbitrage import quote_arbitrage, summarize  # noqa: E402
from rnd.bootstrap import total_uncertainty  # noqa: E402
from rnd.io.cboe import chain_from_table, parse_cboe_delayed  # noqa: E402

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--expiry")
    ap.add_argument("--file", help="use a saved JSON payload instead of downloading")
    ap.add_argument("--save", help="save the downloaded payload here (kept local, not committed)")
    ap.add_argument("--boot", type=int, default=100)
    a = ap.parse_args()
    if a.file:
        payload = json.load(open(a.file))
        qt = pd.Timestamp(payload.get("timestamp")).tz_localize("UTC").tz_convert("America/New_York")
    else:
        import requests
        r = requests.get(URL, timeout=60, headers={"User-Agent": "option-implied-density (research)"})
        r.raise_for_status()
        payload = r.json()
        qt = pd.Timestamp.now(tz="America/New_York")
        if a.save:
            json.dump(payload, open(a.save, "w"))
    df, meta = parse_cboe_delayed(payload)
    spxw = df[df["root"] == "SPXW"]
    exps = sorted(e for e in spxw["expiry"].unique() if pd.Timestamp(f"{e} 16:00").tz_localize("America/New_York") > qt)
    exp = pd.Timestamp(a.expiry).date() if a.expiry else exps[0]
    chain = chain_from_table(df, exp, qt)
    tu = total_uncertainty(chain, n_boot=a.boot)
    base = tu.ensembles[tu.best].base
    arb = summarize(quote_arbitrage(chain, base.fwd))
    print(json.dumps({**{k: v for k, v in base.summary().items()}, **arb, "models_in_band": list(tu.ensembles),
                      "cboe_timestamp": meta.get("cboe_timestamp")}, indent=2, default=float))

    from rnd.plotting import INK_2, S1, style
    import matplotlib.pyplot as plt
    style()
    lo_x, hi_x = base.rnd.quantile([0.002, 0.998])
    x = np.linspace(lo_x, hi_x, 500)
    pt, lo, hi = tu.interval(lambda r: np.maximum(r.pdf(x), 0))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.fill_between(x, lo, hi, color=S1, alpha=0.18, lw=0, label="90% band (quotes + model)")
    ax.plot(x, pt, color=S1, label=f"Risk-neutral density ({tu.best})")
    ax.axvline(base.fwd.F, color=INK_2, lw=1, ls=":", label=f"Parity forward {base.fwd.F:,.1f}")
    ax.set_yticks([])
    ax.set_xlabel("S&P 500 at settlement")
    ax.set_title(f"SPXW {exp} - quotes as of {meta.get('cboe_timestamp')} (Cboe delayed)", loc="left")
    ax.legend(fontsize=8)
    out = ROOT / "results" / "figures" / f"spxw_{exp}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


if __name__ == "__main__":
    main()
