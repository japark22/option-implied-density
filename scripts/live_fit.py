"""Fit real SPXW option chains from Cboe's public delayed quotes and log the results.

Run once per trading day (GitHub Actions, .github/workflows/live.yml) or by hand:

    python scripts/live_fit.py                    # download the current delayed chain
    python scripts/live_fit.py --file chain.json  # reuse a saved payload

For the front (nearest PM-settled) expiry and the expiry closest to one week out:

* the chain is fitted at its *as-of* time (newest trade in the payload), not the
  download time, so time to expiry is measured from when the quotes were valid;
* the density gets a total-uncertainty band (bid-ask bootstrap x two smile models);
* raw quotes are audited for executable (hard) and mid-only (soft) static arbitrage.

Outputs (derived statistics only - raw quotes are never written):
    results/live/history.csv          one row per (as-of time, expiry), de-duplicated
    results/live/latest_<label>.png   smile fit vs market bid/ask IVs, and density with band
    results/live/latest.json          the rows of the latest run
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
from rnd.io.cboe import asof_time, chain_from_table, parse_cboe_delayed, settlement_time  # noqa: E402

URL = "https://cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json"
ET = "America/New_York"
OUT = ROOT / "results" / "live"


def download() -> dict:
    import requests
    for attempt in range(3):
        try:
            r = requests.get(URL, timeout=60,
                             headers={"User-Agent": "option-implied-density (academic research)"})
            r.raise_for_status()
            return r.json()
        except Exception:  # noqa: BLE001
            if attempt == 2:
                raise
            import time
            time.sleep(2 ** attempt)


def pick_expiries(df: pd.DataFrame, asof: pd.Timestamp) -> dict:
    spxw = sorted(df.loc[df["root"] == "SPXW", "expiry"].unique())
    live = [e for e in spxw if settlement_time(e, "SPXW") > asof + pd.Timedelta(minutes=30)]
    if not live:
        return {}
    out = {"front": live[0]}
    target = (asof + pd.Timedelta(days=7)).date()
    week = min(live, key=lambda e: abs((pd.Timestamp(e) - pd.Timestamp(target)).days))
    if week != live[0]:
        out["week"] = week
    return out


def fit_one(df, expiry, asof, n_boot, seed):
    chain = chain_from_table(df, expiry, asof, root="SPXW")
    tu = total_uncertainty(chain, n_boot=n_boot, seed=seed)
    base = tu.ensembles[tu.best].base
    s = base.summary()
    arb = summarize(quote_arbitrage(chain, base.fwd))
    F = base.fwd.F
    q = base.rnd.quantile([0.01, 0.05, 0.5, 0.95, 0.99])
    p_dn, p_dn_lo, p_dn_hi = tu.interval(lambda r: r.prob_below(0.98 * F))
    p_up, p_up_lo, p_up_hi = tu.interval(lambda r: r.prob_above(1.02 * F))
    row = dict(
        asof_et=asof.strftime("%Y-%m-%d %H:%M:%S"), expiry=str(expiry),
        hours_to_expiry=round(chain.T * 365 * 24, 3),
        n_strikes_quoted=int(len(chain.strikes)), n_otm_used=s["n_quotes"],
        F=s["F"], DF=s["DF"], atm_iv=s["atm_iv"],
        best_model=tu.best, models_in_band="+".join(tu.ensembles),
        rmse_halfspreads_svi=tu.ensembles["svi"].base.fit.rmse_halfspreads if "svi" in tu.ensembles else np.nan,
        rmse_halfspreads_spline=(tu.ensembles["spline"].base.fit.rmse_halfspreads
                                 if "spline" in tu.ensembles else np.nan),
        frac_inside_spread=s["frac_inside_spread"],
        q01=q[0], q05=q[1], q50=q[2], q95=q[3], q99=q[4],
        std=s["m_std"], skew=s["m_skew"], kurt=s["m_kurt"],
        p_down_2pct=p_dn, p_down_2pct_lo=p_dn_lo, p_down_2pct_hi=p_dn_hi,
        p_up_2pct=p_up, p_up_2pct_lo=p_up_lo, p_up_2pct_hi=p_up_hi,
        hard_arbitrage=arb["n_hard"], soft_arbitrage=arb["n_soft"], max_hard_arbitrage=arb["max_hard"],
        negative_mass=s["negative_mass"], martingale_error=s["martingale_error"],
        bootstrap_failures=int(sum(e.n_failed for e in tu.ensembles.values())),
    )
    return row, chain, tu, base


def figure(label, row, chain, tu, base, path):
    import matplotlib.pyplot as plt

    from rnd.plotting import INK, INK_2, S1, S2, style

    style()
    otm = base.otm
    T = chain.T
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    k = otm["k"].to_numpy()
    iv_b = np.sqrt(np.clip(otm["w_bid"].to_numpy(), 0, None) / T)
    iv_a = np.sqrt(np.clip(otm["w_ask"].to_numpy(), 0, None) / T)
    a1.vlines(k, iv_b, iv_a, color=INK_2, lw=1.2, label="Market bid-ask (implied vol)")
    kk = np.linspace(k.min(), k.max(), 400)
    colors = {"svi": S2, "spline": S1}
    for m, e in tu.ensembles.items():
        sm = e.base.fit.smile
        a1.plot(kk, sm.iv(kk, T), color=colors.get(m, INK), lw=1.6,
                label=f"{m.upper() if m == 'svi' else 'Spline'} ({e.base.fit.rmse_halfspreads:.2f} half-spreads RMSE)")
    a1.set_xlabel("log-moneyness  ln(K / F)")
    a1.set_ylabel("Implied volatility")
    a1.set_title("Smile: fitted models vs quoted bid-ask", loc="left")
    a1.legend(fontsize=7.5)

    lo_x, hi_x = base.rnd.quantile([0.003, 0.997])
    x = np.linspace(lo_x, hi_x, 500)
    pt, lo, hi = tu.interval(lambda r: np.maximum(r.pdf(x), 0))
    a2.fill_between(x, lo, hi, color=S1, alpha=0.2, lw=0, label="Band (bid-ask bootstrap x models)")
    a2.plot(x, pt, color=S1, label=f"Risk-neutral density ({tu.best})")
    a2.axvline(row["F"], color=INK_2, lw=1, ls=":", label=f"Parity forward {row['F']:,.1f}")
    a2.set_yticks([])
    a2.set_xlabel("S&P 500 at settlement")
    a2.set_title("Option-implied distribution", loc="left")
    a2.legend(fontsize=7.5)
    fig.suptitle(f"SPXW {row['expiry']} ({label}), quotes as of {row['asof_et']} ET, "
                 f"{row['hours_to_expiry']:.1f} h to settlement - hard arbitrage in raw quotes: {row['hard_arbitrage']}",
                 x=0.01, ha="left", fontsize=10.5, fontweight="bold", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file")
    ap.add_argument("--boot", type=int, default=100)
    a = ap.parse_args()
    payload = json.load(open(a.file)) if a.file else download()
    fetched = pd.Timestamp.now(tz="UTC")
    if a.file and payload.get("timestamp"):
        fetched = pd.Timestamp(payload["timestamp"]).tz_localize("UTC")
    df, meta = parse_cboe_delayed(payload)
    if df.empty:
        raise SystemExit("empty payload")
    asof = asof_time(df, fetched)
    OUT.mkdir(parents=True, exist_ok=True)
    hist_p = OUT / "history.csv"
    hist = pd.read_csv(hist_p) if hist_p.exists() else pd.DataFrame()
    rows, errors = [], []
    for label, exp in pick_expiries(df, asof).items():
        if len(hist) and ((hist["asof_et"] == asof.strftime("%Y-%m-%d %H:%M:%S")) &
                          (hist["expiry"] == str(exp))).any():
            print(f"{label} {exp}: already logged for as-of {asof}, skipping")
            continue
        try:
            row, chain, tu, base = fit_one(df, exp, asof, a.boot, seed=int(asof.timestamp()) % 10_000)
            row["label"] = label
            figure(label, row, chain, tu, base, OUT / f"latest_{label}.png")
            rows.append(row)
        except Exception as e:  # noqa: BLE001 - record and continue with the other expiry
            errors.append(f"{label} {exp}: {type(e).__name__}: {e}")
    if rows:
        new = pd.DataFrame(rows)
        hist = pd.concat([hist, new], ignore_index=True) if len(hist) else new
        hist.to_csv(hist_p, index=False, float_format="%.6g")
        (OUT / "latest.json").write_text(json.dumps(rows, indent=2, default=float))
    print(json.dumps({"asof_et": str(asof), "fitted": [r["expiry"] for r in rows], "errors": errors},
                     indent=2))
    if errors and not rows:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
