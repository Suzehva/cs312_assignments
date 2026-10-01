# P1(a): learning from small training budgets

I used the nine supplied d8 runs: peak LR `{0.0015, 0.003, 0.006}` at
`{153.6M, 307.2M, 614.4M}` training tokens. All runs use the same model, so
this part scales the training budget `D`, not model size.

![P1(a) loss--LR fits and scaling rule](plots/p1a_lr_fits_and_scaling.png)

For each budget I fit loss as a quadratic in log LR. Define

```text
x = log2(eta / 0.003),        L_D(eta) = a_D x^2 + b_D x + c_D.
```

This makes the three sampled LRs exactly `x = {-1, 0, 1}`. A simple power law
from LR to loss would be monotone and could not represent the observed
U-shaped curve. The fitted optimum is `x* = -b_D/(2a_D)` and
`eta* = 0.003 * 2^(x*)`.

| Tokens `D` | `a_D` | `b_D` | `c_D` | Fitted `eta*` | Fitted loss |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 153.6M | 0.045784 | 0.030720 | 3.229134 | 0.002378 | 3.223981 |
| 307.2M | 0.014959 | 0.000310 | 3.056223 | 0.002979 | 3.056222 |
| 614.4M | 0.018412 | -0.003297 | 2.925636 | 0.003192 | 2.925488 |

I then fit the three optima with the scale-free rule

```text
eta*(D) = eta_ref (D / D_ref)^beta
        = 0.003276 (D / 614.4M)^0.2125.
```

| Power-law constant | Value |
| --- | ---: |
| `D_ref` | 614.4M tokens |
| `eta_ref` | 0.003276 |
| exponent `beta` | 0.2125 |
| LR multiplier per 2x tokens, `2^beta` | 1.159x |

The fitted optimum increases from 0.00238 to 0.00319 as the budget grows 4x:
longer training prefers a mildly larger peak LR. The exact optima are uncertain
because each loss curve has only three points; fitting quadratics in raw LR
changes their levels but gives nearly the same exponent (0.221 instead of
0.213). P1(b) is the held-out test of whether this proposed rule continues.

**Takeaway.** At fixed d8 model size, doubling the training tokens calls for
about a 16% larger peak learning rate, but the rule is based on only three
budgets and should be treated as a prediction rather than a law.
