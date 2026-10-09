"""Known-answer study: can the pipeline recover a density we know?

For four worlds with an exact terminal density (lognormal, Heston, a bimodal
"event" mixture and a fat-tailed 0DTE-like mixture) we

1. generate realistic quoted chains (tick sizes, spreads, fair value placed
   randomly inside the spread, zero-bid wings) under many noise seeds,
2. run the full pipeline (parity forward -> OTM -> SVI / spline / auto),
3. score the recovered density against the truth (L1 density error, KS CDF
   error, mean absolute error on 25-point buckets - the Kalshi bucket width),
4. measure whether the 90% bid-ask bootstrap band actually covers the true
   bucket probabilities 90% of the time.

Every number written to results/ is computed by this script.

Usage:  python scripts/known_answer_study.py [--reps 30] [--cov-reps 12] [--boot 60]
"""
from __future__ import annotations

import argparse
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from rnd import fit_rnd  # noqa: E402
from rnd.bootstrap import bootstrap_rnd, total_uncertainty  # noqa: E402
from rnd.synthetic import Heston, Lognormal, LognormalMixture, quote_chain  # noqa: E402

F = 6700.0
BUCKET = 25.0


def cases():
    T30, T7, T0 = 30 / 365, 7 / 365, 5 / (24 * 365)
    return {
        "lognormal_30d": dict(model=Lognormal(F, 0.18, T30), T=T30, K=np.arange(4500, 8500, 25.0),
                              DF=0.995, rel_spread=0.01, label="Lognormal, 30 days"),
        "heston_30d": dict(model=Heston(F, T30), T=T30, K=np.arange(4500, 8000, 25.0),
                           DF=0.995, rel_spread=0.01, label="Heston (skew), 30 days"),
        "event_bimodal_7d": dict(model=LognormalMixture(F, (0.5, 0.5), (0.97, 1.03), (0.10, 0.10), T7),
                                 T=T7, K=np.arange(5800, 7600, 10.0), DF=0.998, rel_spread=0.01,
                                 label="Bimodal event, 7 days"),
        "fat_tail_0dte": dict(model=LognormalMixture(F, (0.8, 0.2), (1, 1), (0.12, 0.30), T0), T=T0,
                              K=np.arange(6550, 6850, 5.0), DF=1.0, rel_spread=0.02,
                              label="Fat-tailed 0DTE, 5 hours"),
    }


def bucket_edges(model, lo_q=0.005, hi_q=0.995):
    x = np.linspace(F * 0.5, F * 1.5, 20001)
    c = model.cdf(x)
    lo, hi = np.interp([lo_q, hi_q], c, x)
    return np.arange(np.floor(lo / BUCKET) * BUCKET, np.ceil(hi / BUCKET) * BUCKET + BUCKET, BUCKET)


def score(rnd, model, edges):
    x = rnd.grid
    l1 = float(np.trapezoid(np.abs(rnd.pdf_clean - model.pdf(x)), x))
    ks = float(np.max(np.abs(rnd.cdf(x) - model.cdf(x))))
    est = rnd.prob_between(edges[:-1], edges[1:])
    true = model.cdf(edges[1:]) - model.cdf(edges[:-1])
    return l1, ks, float(np.mean(np.abs(est - true))), float(np.max(np.abs(est - true)))


def run_point(args):
    name, seed = args
    c = cases()[name]
    ch = quote_chain(c["model"], c["K"], c["T"], DF=c["DF"], rel_spread=c["rel_spread"], seed=seed)
    edges = bucket_edges(c["model"])
    rows = []
    for method in ("svi", "spline", "auto"):
        t = time.time()
        try:
            r = fit_rnd(ch, method)
            l1, ks, mae, mx = score(r.rnd, c["model"], edges)
            rows.append(dict(case=name, seed=seed, method=method, chosen=r.fit.method,
                             l1=l1, ks=ks, bucket_mae=mae, bucket_max=mx,
                             rmse_halfspreads=r.fit.rmse_halfspreads,
                             frac_inside=r.fit.frac_inside_spread,
                             F_err_bp=1e4 * (r.fwd.F / F - 1), seconds=time.time() - t, error=""))
        except Exception as e:  # noqa: BLE001 - record, never hide
            rows.append(dict(case=name, seed=seed, method=method, error=str(e)))
    return rows


def run_coverage(args):
    name, seed, n_boot = args
    c = cases()[name]
    ch = quote_chain(c["model"], c["K"], c["T"], DF=c["DF"], rel_spread=c["rel_spread"], seed=1000 + seed)
    edges = bucket_edges(c["model"])
    true = c["model"].cdf(edges[1:]) - c["model"].cdf(edges[:-1])
    material = true > 0.01                       # buckets a trader would care about
    tu = total_uncertainty(ch, n_boot=n_boot, seed=seed, grid_n=801)
    row = dict(case=name, seed=seed, n_buckets=int(material.sum()), best=tu.best)
    bands = {m: e.prob_between(edges[:-1], edges[1:]) for m, e in tu.ensembles.items()}
    bands["union"] = tu.prob_between(edges[:-1], edges[1:])
    for m, (pt, lo, hi) in bands.items():
        cov = (true >= lo - 1e-12) & (true <= hi + 1e-12)
        row[f"cov_{m}"] = float(cov[material].mean())
        row[f"width_{m}"] = float(np.mean((hi - lo)[material]))
        row[f"abserr_{m}"] = float(np.mean(np.abs(pt - true)[material]))
    # residual error beyond the union band, used as a model-error tolerance downstream
    pt, lo, hi = bands["union"]
    excess = np.maximum(np.maximum(lo - true, true - hi), 0.0)[material]
    row["excess_p90"] = float(np.percentile(excess, 90))
    row["excess_max"] = float(excess.max())
    return row


def hero_figure(path):
    from rnd.plotting import INK, S1, S2, style
    import matplotlib.pyplot as plt

    style()
    cs = cases()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.6))
    for ax, (name, c) in zip(axes.ravel(), cs.items()):
        ch = quote_chain(c["model"], c["K"], c["T"], DF=c["DF"], rel_spread=c["rel_spread"], seed=7)
        ens = bootstrap_rnd(ch, n_boot=60, method="auto", seed=7, grid_n=801)
        svi = fit_rnd(ch, "svi")
        lo_x, hi_x = np.interp([0.002, 0.998], c["model"].cdf(np.linspace(F * .5, F * 1.5, 20001)),
                               np.linspace(F * .5, F * 1.5, 20001))
        x = np.linspace(lo_x, hi_x, 600)
        pt, lo, hi = ens.pdf_band(x)
        ax.fill_between(x, lo, hi, color=S1, alpha=0.18, lw=0, label="90% bid-ask bootstrap band")
        ax.plot(x, c["model"].pdf(x), color=INK, lw=1.4, ls=(0, (4, 3)), label="True density")
        ax.plot(x, pt, color=S1, lw=2, label=f"Recovered (auto-selected: {ens.base.fit.method})")
        if ens.base.fit.method != "svi":
            rm = svi.fit.rmse_halfspreads
            tag = f"rejected, pricing RMSE {rm:.0f} half-spreads" if rm > 2 else f"alternative, RMSE {rm:.1f} half-spreads"
            ax.plot(x, np.maximum(svi.rnd.pdf(x), 0), color=S2, lw=1.2, label=f"SVI ({tag})")
        ax.set_title(c["label"])
        ax.set_yticks([])
        ax.set_xlabel("S&P 500 level at expiry")
        ax.legend(fontsize=7.5, loc="upper left")
    fig.suptitle("Known-answer test: risk-neutral density recovered from noisy bid/ask quotes",
                 fontsize=12, fontweight="bold", color=INK, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--cov-reps", type=int, default=12)
    ap.add_argument("--boot", type=int, default=60)
    ap.add_argument("--procs", type=int, default=2)
    a = ap.parse_args()
    out = ROOT / "results"
    (out / "figures").mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    names = list(cases())
    with Pool(a.procs) as pool:
        rows = [r for rs in pool.map(run_point, [(n, s) for n in names for s in range(a.reps)]) for r in rs]
        cov = pool.map(run_coverage, [(n, s, a.boot) for n in names for s in range(a.cov_reps)])
    runs = pd.DataFrame(rows)
    runs.to_csv(out / "known_answer_runs.csv", index=False)
    cov = pd.DataFrame(cov)
    cov.to_csv(out / "known_answer_coverage.csv", index=False)

    ok = runs[runs["error"].fillna("") == ""]
    summ = ok.groupby(["case", "method"]).agg(
        n=("seed", "size"), l1=("l1", "mean"), ks=("ks", "mean"), bucket_mae_pp=("bucket_mae", "mean"),
        bucket_max_pp=("bucket_max", "mean"), rmse_hs=("rmse_halfspreads", "median"),
        inside=("frac_inside", "mean"), F_err_bp=("F_err_bp", lambda s: np.mean(np.abs(s))),
        chose_spline=("chosen", lambda s: np.mean(s == "spline"))).reset_index()
    summ["bucket_mae_pp"] *= 100
    summ["bucket_max_pp"] *= 100
    summ["failures"] = summ.apply(lambda r: int(((runs["case"] == r["case"]) & (runs["method"] == r["method"])
                                                 & (runs["error"].fillna("") != "")).sum()), axis=1)
    summ.to_csv(out / "known_answer_summary.csv", index=False)
    agg = {"reps": ("seed", "size")}
    for m in ("svi", "spline", "union"):
        if f"cov_{m}" not in cov:
            continue
        agg[f"coverage_{m}"] = (f"cov_{m}", "mean")
        agg[f"width_{m}_pp"] = (f"width_{m}", "mean")
    agg["excess_p90_pp"] = ("excess_p90", "mean")
    agg["excess_max_pp"] = ("excess_max", "max")
    covs = cov.groupby("case").agg(**agg).reset_index()
    for col in covs.columns:
        if col.endswith("_pp"):
            covs[col] *= 100
    covs.to_csv(out / "known_answer_coverage_summary.csv", index=False)

    hero_figure(out / "figures" / "known_answer_hero.png")
    with open(out / "known_answer_summary.md", "w") as f:
        f.write(f"Generated by scripts/known_answer_study.py (reps={a.reps}, coverage reps={a.cov_reps}, "
                f"bootstrap draws={a.boot}). Runtime {time.time() - t0:.0f}s.\n\n")
        f.write(summ.round(4).to_markdown(index=False))
        f.write("\n\n90% bootstrap band coverage of true 25-pt bucket probabilities (buckets with p > 1%):\n\n")
        f.write(covs.round(3).to_markdown(index=False))
        f.write("\n")
    print(summ.round(4).to_string(index=False))
    print(covs.round(3).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
