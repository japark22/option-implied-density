# Methodology

## Pipeline

```
quoted chain (bid/ask, calls + puts, one expiry)
  │
  ├─ 1. Parity forward      C_mid − P_mid = DF·(F − K), weighted LS on the 20 strikes nearest ATM,
  │                         one 3-MAD outlier pass. No rate or dividend input. DF fixed at 1 under 3 days.
  ├─ 2. OTM table           puts below F, calls at/above; walk outward and stop after two consecutive
  │                         zero bids (the Cboe VIX truncation rule); undiscounted prices → total variance w.
  ├─ 3. Smile fit           SVI (raw, 5 params, multi-start, min-variance and Lee-slope penalties)
  │                         or penalised cubic spline in w (vega/half-spread weights).
  │                         Both minimise pricing error measured in half-spreads.
  ├─ 4. Density             analytic: q(K) = g(k)·φ(d₂) / (K√w),  CDF(K) = 1 − N(d₂) + F·φ(d₁)·w'(k) / (2K√w)
  ├─ 5. Uncertainty         bootstrap: redraw each two-sided quote ~ U[bid, ask], rerun 1–4
  │                         (model choice and spline penalty frozen); union across fitters that price the
  │                         quotes (RMSE ≤ max(2, 1.5× best) half-spreads).
  └─ 6. Diagnostics         raw-quote static arbitrage (hard = survives bid/ask; soft = mid only), Durrleman g(k),
                            negative mass, tail mass beyond the grid, martingale error E[S] − F.
```

## Choices that matter

**Differentiate the fit, not the quotes.** Second differences of quoted prices amplify half-a-tick noise by 1/ΔK². We differentiate the fitted smile in closed form, using Gatheral's butterfly function g(k). There is no step size to tune, and a negative density shows up as g < 0, which is reported rather than smoothed away.

**Error in half-spreads.** Both fitters weight pricing errors by the quoted half-spread, so "fit quality" has a market meaning: an RMSE of 1 half-spread is as good as the quotes allow.

**Choosing the spline penalty with the discrepancy principle.** The penalty is the largest λ whose RMS pricing error is within 1/√3 half-spreads. That is the RMS distance from mid of a fair value spread uniformly over [bid, ask]. The result is the smoothest curve the quotes cannot reject.

**Spline wings.** Beyond the last quote, the wing's curvature decays as $(u_2 + βd)e^{−ad}$ from the edge value. This keeps w, w′ and w″ continuous, so the density has no jump at the last strike. β is solved in closed form so that the asymptotic slope lands inside Lee's moment bound [0, 2].

**Why two models.** SVI is parametric and smooth, with good wings. It cannot represent the W-shaped smile of a bimodal event distribution. The spline can, at the cost of noisier tails. Comparing the two is the cheapest honest measure of model risk. The known-answer study shows that bootstrap bands from a *single* misspecified model are badly overconfident.

**Bootstrap with frozen tuning.** If the spline penalty is re-selected inside each replicate, it jumps between grid values. The bands then fill with tuning noise rather than quote noise. Freezing model choice and λ at their values on the observed quotes isolates quote uncertainty.

## Known-answer study

`scripts/known_answer_study.py` runs four worlds with exact densities:

| Case | Model | Why |
|---|---|---|
| `lognormal_30d` | Black-Scholes, σ = 18% | sanity: the trivial case must be near-perfect |
| `heston_30d` | Heston (v₀ = θ = 0.04, κ = 1.5, ξ = 0.6, ρ = −0.7) | equity skew; density by Fourier inversion of the CF |
| `event_bimodal_7d` | 50/50 lognormal mixture at ±3% | an FOMC/CPI-style binary event |
| `fat_tail_0dte` | 80/20 mixture, σ = 12% / 30%, 5 hours | the Kalshi use case: same-day expiry, fat tails |

Quotes follow SPX tick rules: $0.05 below $3 and $0.10 above. Spreads are max($0.10, 1–2% of premium). Fair value is placed uniformly inside the spread rather than at mid, and options worth less than a tick are quoted zero-bid. Errors are reported as:

- L1 distance between densities, ∫|q̂ − q|;
- KS distance between CDFs;
- mean and max absolute error on 25-point buckets (the Kalshi range width);
- coverage of the 90% band for buckets with true probability above 1%.

## Validation tools (`rnd.validation`)

- **PIT and Berkowitz (2001)** likelihood-ratio test (mean, variance, AR(1)) for density forecasts.
- **Diebold-Mariano** test with Newey-West variance, for comparing two forecasters on identical outcomes.
- **Log score, Brier score and reliability tables.**

## References

- Breeden, D. & Litzenberger, R. (1978). Prices of state-contingent claims implicit in option prices. *Journal of Business*.
- Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization (SVI).
- Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces. *Quantitative Finance*.
- Lee, R. (2004). The moment formula for implied volatility at extreme strikes. *Mathematical Finance*.
- Figlewski, S. (2010). Estimating the implied risk-neutral density for the US market portfolio.
- Albrecher, H. et al. (2007). The little Heston trap.
- Berkowitz, J. (2001). Testing density forecasts. *Journal of Business & Economic Statistics*.
- Diebold, F. & Mariano, R. (1995). Comparing predictive accuracy. *JBES*.
- Cboe VIX White Paper (strike truncation after two consecutive zero bids).
