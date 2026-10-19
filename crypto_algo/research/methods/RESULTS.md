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

## Structure batch on the best bot (research/methods/structure.py), October 2026

11 ideas (universe, sizing, ranking) on top of the aggressive volume-profile bot; protocol in the module docstring.
Passed all pre-lock tests: monthly universe refresh, inverse-vol sizing 0.7. Greedy train combination = both.
One look at the locked year (2025-10-01..now), rule fixed before: the combination's median Sharpe must beat the current bot.

| model | train Sharpe / CAGR / DD | 2023-25.09 Sharpe / CAGR / DD | locked year Sharpe / return / DD |
|---|---|---|---|
| current best | 1.83 / +97% / -24% | 1.10 / +33% / -28% | 1.39 / +31% / -13% |
| monthly refresh | 1.89 / +96% / -24% | 1.16 / +35% / -29% | 1.49 / +41% / -11% |
| inverse-vol 0.7 | 1.89 / +62% / -17% | 1.09 / +27% / -26% | 1.18 / +22% / -12% |
| both (declared candidate) | 1.91 / +63% / -17% | 1.13 / +28% / -23% | 1.22 / +26% / -10% |

Decision: the declared combination fails, so the current bot is kept. Monthly refresh alone was better on every
period including the locked year, but that is an after-the-fact reading of the one look; it needs forward (paper) evidence.

## Geometry batch (research/methods/geometry.py), October 2026

10 ideas on exits, channel lengths, trailing stops and the daily confirmation. Locked year not used.
Passed all tests: faster breakout channels 20/50/100 (train Sharpe 1.91 vs 1.83, +105% vs +97%/yr, 9/9 variants,
A +0.78 vs +0.70, B -0.16 vs -0.22, neighbour 30/75/150 also better). Added to the paper demo as `best_fast_entries`
(paper-v4), not to the bot. Failed: exits 5/10 and 20/40, entries 100/200/400, 3 and 5 ATR trailing stops,
daily EMA20 gate, higher-lows gate, extra weekly gate, exit on daily EMA20.
