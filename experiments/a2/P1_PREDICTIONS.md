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

## P1(c): two source-fit predictions

Recorded on 2026-10-01 at 05:21 UTC (September 30, Los Angeles), using
only the supplied P1(a) and P1(b) measurements. No P1(c) target losses
have been inspected.

For each source budget, estimate the optimal LR with the same log-LR
quadratic as in P1(a,b). Fit a power law to these optima by unweighted
least squares in log tokens versus log optimal LR.

```text
All six budgets:
eta*(D) = 0.00348168553639 * (D / 2457600000)^0.0988099083267
eta*(4915200000) = 0.00372850122083

Only the three larger budgets (1.2288B, 1.8432B, 2.4576B):
eta*(D) = 0.00295665882843 * (D / 2457600000)^(-0.409276680985)
eta*(4915200000) = 0.00222636648823
```

Initial expectation: the all-six-budget fit will give lower target loss
because more source optima may reduce variance. The competing argument
is that the three larger budgets capture the recent downturn, so they
may have less extrapolation bias. The target measurements will test this.

The target runs at these two predicted LRs have not been launched by this
analysis. The fixed target grid remains 0.0015, 0.003, and 0.006.

## P1(d): Hyperball target prediction

Recorded on 2026-10-01 at 05:55 UTC (September 30, Los Angeles), before
launching a Hyperball target run or inspecting any Hyperball target losses.

Use only the 24 supplied P1(d) runs at 153.6M, 307.2M, and 614.4M tokens.
Fit a quadratic in log LR to all eight LRs per budget, then fit a power law
to the three inferred optima:

```text
eta*(D) = 0.010267261603103506 * (D / 614400000)^(-0.2918504903975972)
eta*(1228800000) = 0.008386850003371452
```

The recorded target peak LR is **0.00838685** (Hyperball's matrix-group LR).
This uses the primary all-eight-LR fits, not the local-four-LR sensitivity
check. The prediction extrapolates to twice the largest source budget;
its accuracy is not yet tested. No Hyperball target job has been launched
by this analysis.
