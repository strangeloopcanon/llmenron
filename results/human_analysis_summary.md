# Human Topic-Juggling Results (Enron)

- Messages analyzed: **516,796**
- Custodians analyzed: **150**
- Inferential sample (>= 200 messages, >=20 active days): **148 custodians**
- Median mean active threads (14-day rolling): **49.97**
- Median 90th percentile active threads: **104.50**

## Rank/Success Proxy Effect
- Spearman rho(seniority score, mean active threads): **-0.014** (p=0.8631)
- OLS coefficient for seniority score (controlling for log total messages): **-1.343** (95% CI -4.032 to 1.345, p=0.3251)
- ANOVA p-value for seniority tiers (controlling for log volume): **0.2645**

## What Actually Predicts Juggling
- Spearman rho(total messages, mean active threads): **0.573** (p=2.85e-14)
- Spearman rho(total messages, p90 active threads): **0.549** (p=5.28e-13)

## Human-Derived N Levels For Next Experiment
- Mean-active quantiles (q25/q50/q75/q90): **34.2 / 50.3 / 68.9 / 97.9**
- Stress (p90-active) quantiles (q25/q50/q75/q90): **72.8 / 105.3 / 143.4 / 222.3**
