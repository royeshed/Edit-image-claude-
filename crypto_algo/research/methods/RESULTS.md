# Method research, October 2026: results

Protocol: research/harness.py. Each family was researched on 2018-11..2022 only (at most 20 trials,
median of 9 universe variants, must beat the current bot in at least 6 of 9 variants), then reviewed by
an independent agent: look-ahead (code read plus `lab.leak_check`), constraints, reproduction and
perturbations. 2023-01..2026-10 stayed locked until one final run (`research/methods/stack.py lockbox`),
with the adoption rule fixed in advance. About 200 trials are logged in data/methods/trials.jsonl,
counting the reviewers' trials.

| family | train verdict | review | frozen |
|---|---|---|---|
| sizing | improvement | pass | inverse-vol slots, ivol 0.7, k=3 |
| regime | improvement | pass with caveats | BTC weekly close > weekly EMA20 |
| exits | improvement | pass with caveats | Donchian exits 5/10 instead of 10/20 |
| signal_ensemble | none (EMA cross, TSMOM, Keltner, MACD, Supertrend, blends all worse) | pass with caveats | - |
| timeframes | none (daily + weekly sleeves, 12h-equivalent sleeve, blends) | pass | - |
| ranking | none (4-56d momentum, risk-adjusted, vs BTC, trend quality, volume surge) | pass | - |
| turnover | none (hysteresis, min hold, bands, daily decisions) | pass with caveats | - |
| universe_composite | none (quality composites, correlation-aware, ATH filter, age) | pass | - |
| universe_ml | none (walk-forward ridge / LightGBM / HGB, hybrids) | pass | - |

## Lockbox (median of the 9 variants)

| model | train Sharpe | train CAGR | train maxDD | 2023-26 Sharpe | 2023-26 CAGR | 2023-26 maxDD | beats current 2023-26 |
|---|---|---|---|---|---|---|---|
| current bot | 1.57 | +52% | -26% | 0.81 | +18% | -28% | - |
| R regime | 1.75 | +57% | -24% | 0.54 | +10% | -29% | 0/9 |
| X exits 5/10 | 1.65 | +52% | -28% | 0.86 | +16% | -26% | 5/9 |
| Z inverse-vol 0.7 | 1.78 | +39% | -17% | 0.86 | +15% | -22% | 9/9 |
| R+X+Z0.7 (primary) | 2.12 | +37% | -12% | 0.54 | +8% | -24% | 1/9 |
| R+X | 1.85 | +56% | -18% | 0.53 | +10% | -26% | 0/9 |
| R+X+Z1.0 k4 | 1.88 | +58% | -18% | 0.70 | +14% | -29% | 1/9 |

Decision (rule fixed before opening): the primary stack fails, so we keep the current bot.

What we learned:
- The BTC weekly regime filter fit two train bear phases (mid-2021, 2022). In 2023 it kept the bot out
  of the recovery (+0% vs +19%) and in 2026 it cut participation (+6% vs +18%).
- Inverse-volatility sizing is the only component that held up in every variant out of sample. It is a
  risk dial (lower drawdown, lower CAGR), not extra return. It was picked after the lockbox, so adopting
  it would need forward (paper) evidence.
- Simple beat complex: the current Donchian / daily-gate / top-3-momentum bot survived nine families
  of alternatives.
