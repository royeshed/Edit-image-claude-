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

## Market-ideas batch (research/methods/market_ideas.py), October 2026

9 ideas taken from well-known systems (Connors RSI-2 dip sleeve, Turtle pyramiding, Faber 200-day BTC/ETH core,
Fear & Greed filter, weekend breakouts, same-hour RVOL, strong-close breakout bar, not-overextended, Freqtrade-style
cooldown after a loss). The data was cut at 2025-10-01 before loading (Lab.truncate), so the locked year was not read.
Baseline reproduced: train Sharpe 1.83 / +97% / -24%, A +0.70, B -0.22. Leak check 0.0.

| idea | train Sh / CAGR / DD | beats | A | B | plateau |
|---|---|---|---|---|---|
| dip_sleeve_rsi2 | 1.41 / +73% / -35% | 0/9 | +0.24 | -0.60 | no |
| pyramid_after_1atr | 1.71 / +86% / -24% | 0/9 | +0.60 | -0.09 | no |
| core_btc_eth_200d | 1.75 / +101% / -29% | 0/9 | +1.02 | +0.30 | no |
| fear_greed_below_80 | 1.78 / +69% / -24% | 1/9 | +0.59 | -0.38 | no |
| skip_weekend_breakouts | 2.01 / +92% / -21% | 9/9 | +0.60 | +0.05 | ok |
| rvol_same_hour | 1.66 / +81% / -30% | 0/9 | +0.61 | +0.01 | no |
| strong_close | 1.87 / +86% / -24% | 3/9 | +0.84 | -0.05 | ok |
| not_overextended | 1.70 / +75% / -26% | 0/9 | +0.33 | +0.28 | no |
| cooldown_after_loss | 1.70 / +78% / -24% | 0/9 | +0.75 | -0.05 | no |

Nothing passed; the bot stays as it is. Near misses, not to be re-tuned on the same data: skipping weekend breakouts
(passed train, B and plateau, failed A 0.60 vs 0.70) and the BTC/ETH 200-day core (worse on train, better on both
holdouts: it helped in the BTC-led 2023-25 market). Both could only be revisited as forward (paper) evidence.

## Costs, deflated Sharpe and new data (October 2026; data cut at 2025-10-01, nothing tuned)

**Costs** (`research/methods/costs.py`, best bot, median of 9 variants). The bot trades 41x (train) / 52x (2023-25.09)
its equity a year, so 0.12% per side costs 4.9% / 6.2% of equity a year.

| scenario | train CAGR / Sharpe | 2023-25.09 CAGR / Sharpe |
|---|---|---|
| now, 0.12% per side | +97.2% / 1.83 | +32.7% / 1.10 |
| fees paid in BNB, 0.095% | +99.5% / 1.85 | +33.8% / 1.14 |
| BNB + limit orders (buys and sells) | +94.2% / 1.83 | +29.8% / 1.03 |
| zero cost (ceiling) | +107.7% / 1.93 | +40.6% / 1.26 |
| now + 5% a year on idle USDT | +105.1% / 1.96 | +38.3% / 1.23 |
| BNB + 5% on idle USDT | +107.8% / 1.98 | +39.4% / 1.26 |

Limit orders at the decision close fill 97% of the time, but the 3% that do not are the breakouts that run away,
and missing them costs more than the spread saved: keep market orders. BNB fees and interest on idle cash both help.

**Deflated Sharpe** (`research/methods/deflated.py`, 15 coins, Q-jan). With 350 trials and a trial spread of 0.15-0.30,
luck alone reaches a Sharpe of 0.44-0.88. Train Sharpe 2.03: DSR 99-100%. 2023-25.09 Sharpe 0.97: DSR 56-81%, so the
recent edge is real but not yet statistically strong. Noise bar for a new idea's train gain: about 0.25-0.5 Sharpe as the
best of 140 ideas (0.36 for the weekend idea, which gained only +0.15). New rule for future ideas: a train gain above
that bar, and pass holdouts A and B, and holdout C (coins 66-85, kept unused for this purpose from now on).

**New data, unchanged bot** (`research/methods/newdata.py`):

| test | bot | comparison |
|---|---|---|
| coins 66-85 (never used), 2018-11..2022 | -1%/yr, Sharpe -0.01, DD -24% | equal-weight hold -28%/yr, DD -85% |
| coins 66-85, 2023-01..2025-09 | +26%/yr, Sharpe 0.79, DD -37% | equal-weight hold +8%/yr, DD -76%; BTC +102%/yr |
| OKX prices (64 coins, 81% coverage), train | +52%, Sharpe 1.46 | same coins on Binance: +68%, 1.65 |
| OKX prices, 2023-01..2025-09 | +31%, Sharpe 0.98 | Binance: +31%, 0.96 |
| 38 ETFs, daily, 2008-2025.09 | +1%/yr, Sharpe 0.26 | SPY +11%/yr 0.61; 60/40 +8% 0.75 |

The edge does not depend on Binance's prints (OKX gives the same 2023-25 result), protects well on small coins but
earns little there, and does not carry over to traditional markets: it is a crypto, large-coin trend effect.
