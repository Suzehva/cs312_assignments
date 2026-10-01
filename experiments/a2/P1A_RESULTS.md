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

I then fit a power law relating the training-token budget `D` to the optimal
learning rate:

```text
eta*(D) = eta_ref (D / D_ref)^beta
        = 0.003276 (D / 614.4M)^0.2125.
```

Longer training prefers a larger peak LR. With only three measurements per
budget, precise minima depend on the fit; quadratics in raw LR give a similar
scaling exponent (0.221). P1(b) tests the prediction.

**Takeaway.** Our fitted rule: double the training data, multiply peak LR
by 1.16. This estimate uses only three budgets.

The quadratic approximates a smooth minimum.
[Hoffmann et al. (2022), Section 3.2](https://arxiv.org/abs/2203.15556)
similarly fit parabolas to find optima, then power laws across budgets, but
their swept variable is model size. This is an analogy for the method,
not evidence for the LR curve. Original figures and their laws are recorded
in the [figure source notes](../../6abdc5b0bad58b38dfd83f81/figures/SOURCES.md).
