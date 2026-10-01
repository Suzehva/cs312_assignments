# Predictions recorded before inspecting held-out losses

## P1(b): predictions from P1(a)

Recorded on 2026-10-01 at 03:44 UTC (September 30, Los Angeles), before
loading P1(b) measurements or inspecting any P1(c) target losses.

Using only the nine P1(a) measurements, the fitted rule is

```text
eta*(D) = 0.00327620951889 * (D / 614400000)^0.212507174509.
```

| Target training tokens | Predicted optimal peak LR |
| ---: | ---: |
| 1,228,800,000 | 0.00379614420429 |
| 1,843,200,000 | 0.00413774152242 |
| 2,457,600,000 | 0.00439859256152 |

The same small-budget rule predicts 0.00509664951620 at 4,915,200,000
tokens. This is a P1(a)-only extrapolation, not either of P1(c)'s required
all-six-budget and larger-three-budget predictions.

The three prescribed P1(c) grid runs use peak LRs 0.0015, 0.003, and
0.006. Their target losses must not be inspected until the two P1(c)
source-fit predictions have been recorded.
