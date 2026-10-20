# The mathematics of `oracle_forecast`

`oracle_forecast` turns realized returns into a forecast whose rank IC is known
in advance, so that portfolio construction and trade calculation can be tested
against a forecast with no unknowns of its own. This note derives everything
the implementation relies on. Throughout, $n$ is the number of names with a
finite realized return in one `(decision, horizon)` cross-section; every
quantity below is recomputed from that $n$ for every cross-section.

## 1. Rank IC as a sum, and why swaps are cheap to cost

Let $r_i$ be the average rank of name $i$'s realized return and $f_i$ the rank
the forecast assigns it. Spearman's rank IC is the Pearson correlation of $f$
with $r$:

$$
\rho = \frac{C}{\sqrt{S_f S_r}},
\qquad
C = \sum_i (f_i - \bar f)(r_i - \bar r),
\quad
S_f = \sum_i (f_i - \bar f)^2,
\quad
S_r = \sum_i (r_i - \bar r)^2 .
$$

The forecast is always a strict ordering, so $f$ is a permutation of
$0, \dots, n-1$ and $S_f = n(n^2-1)/12$ is fixed. The realized returns may
have ties, in which case $r$ carries average ranks, $S_r < S_f$, and the
perfect ranking has $\rho_{\max} = C_{\max}/\sqrt{S_f S_r} < 1$. Everything
that can change is in $C$.

Exchanging the forecast ranks of names $a$ and $b$ changes only two terms of
$C$:

$$
\Delta C = (f_b - \bar f)(r_a - \bar r) + (f_a - \bar f)(r_b - \bar r)
        - (f_a - \bar f)(r_a - \bar r) - (f_b - \bar f)(r_b - \bar r)
        = -(f_a - f_b)(r_a - r_b).
$$

So a swap **lowers $C$ by exactly $(f_a - f_b)(r_a - r_b)$**, evaluated on
the current ranking, in constant time, ties included. The familiar
$\rho = 1 - 6\sum d_i^2 / (n(n^2-1))$ is the special case of no ties, and the
implementation never uses it. Instead it works with a *budget*: to reach a
target $\rho^\star$ it must remove

$$
\text{need} = C_{\max} - \rho^\star \sqrt{S_f S_r}
$$

from $C$, where $C_{\max}$ is the perfect ranking's value.

## 2. Mirror pairs: `tails` and `middle`

Work in realized order, so position $i$ holds the $i$-th smallest return and
the perfect forecast has $f_i = i$. A *mirror pair* exchanges positions $i$
and $n-1-i$. Mirror pairs are disjoint and all start from the perfect ranking,
so their costs are fixed in advance and simply add:

$$
\text{cost}_i = (n-1-2i)\,(r_{n-1-i} - r_i),
$$

which without ties is $(n-1-2i)^2$.

**`tails`** takes pairs in the order $i = 0, 1, 2, \dots$ (best with worst
first). Without ties the first $k$ pairs remove

$$
\sum_{i=0}^{k-1} (n-1-2i)^2
= k(n-1)^2 - 2(n-1)k(k-1) + \tfrac{2}{3}k(k-1)(2k-1),
$$

and the outermost pair alone moves the IC by
$12(n-1)^2 / (n(n^2-1)) \approx 12/n$. That is large: for a cross-section of
a few hundred names one pair is more than the whole IC a real forecast would
have, which is why at a low IC `tails` *inverts* the extremes rather than
scrambling them, and why only a small fraction of names is ever touched.

**`middle`** takes pairs from the median outwards. For even $n$ the distances
are $1, 3, 5, \dots$, so $k$ pairs remove $\sum_{i<k}(2i+1)^2 = k(4k^2-1)/3$
and

$$
\rho(k) = 1 - \frac{4k(4k^2-1)}{n(n^2-1)} \approx 1 - 16\,\frac{k^3}{n^3},
\qquad
k(\rho) \approx n\left(\frac{1-\rho}{16}\right)^{1/3}.
$$

Reaching a small IC therefore disturbs about $n/16^{1/3} \approx 0.4n$ pairs,
i.e. roughly 80% of the names, leaving the outer 20% exactly right. The slope
at that point is $d\rho/dk = 48k^2/n^3 \approx 3\cdot16^{1/3}(1-\rho)^{2/3}/n
\approx 7.6/n$.

**Hitting the target.** Mirror pairs are coarse, so the budget rarely divides
evenly. The implementation takes the longest prefix of the priority order
that fits, then spends the remainder in whichever of two ways lands closer:
adding later pairs that still fit (natural for `tails`, whose later pairs are
smaller), or taking the overshooting pair and handing back earlier pairs,
latest first (natural for `middle`, whose later pairs are larger). Because
every cost is exact, the result is within about $10^{-6}$ of the target IC
without ties, and about $10^{-4}$ with heavy ties, where many pairs cost zero
and the budget becomes lumpy. The side effect is that a "wrong" region is
mostly rather than entirely wrong: a few of the cheapest pairs sit on the
other side of the boundary.

## 3. Random swaps: the expected IC after $k$ transpositions

Pick an ordered pair $(a, b)$, $a \ne b$, uniformly. Summing the swap cost
over all ordered pairs,

$$
\sum_{a \ne b} (f_a - f_b)(r_a - r_b)
= 2n\sum_i f_i r_i - 2\Big(\sum_i f_i\Big)\Big(\sum_i r_i\Big)
= 2n\,C ,
$$

since the diagonal terms vanish. There are $n(n-1)$ ordered pairs, so

$$
\mathbb{E}[\Delta C] = -\frac{2C}{n-1}
\quad\text{whatever the current ranking.}
$$

This is the whole result. One random transposition multiplies the expected
$C$, and hence the expected IC, by $1 - 2/(n-1)$, independent of how
scrambled the ranking already is and of ties.

**With replacement** (a random transposition walk, names may be swapped
again):

$$
\mathbb{E}[\rho_k] = \rho_{\max}\left(1 - \frac{2}{n-1}\right)^k,
\qquad
k = \frac{\ln(\rho_{\max}/\rho^\star)}{-\ln(1 - 2/(n-1))}
\approx \frac{n-1}{2}\ln\frac{\rho_{\max}}{\rho^\star}.
$$

**Without replacement** (disjoint pairs, each name moved at most once): all
costs are evaluated on the perfect ranking. For distinct uniform positions
$a, b$ in $0, \dots, n-1$, $\mathbb{E}[(a-b)^2] = n(n+1)/6$, and
$C_{\max} = n(n^2-1)/12$ without ties, so each pair removes $2/(n-1)$ of
$C_{\max}$ and the decay is linear:

$$
\mathbb{E}[\rho_k] = 1 - \frac{2k}{n-1},
\qquad
k = \frac{(1-\rho^\star)(n-1)}{2}.
$$

The two agree to first order, as they must, since from the perfect ranking
$2C_{\max}/(n-1) = n(n+1)/6$. The disjoint version cannot pass $\rho \approx
0$ (a full matching lands at $-1/(n-1)$), and reaching a low IC with it moves
nearly every name.

**What the implementation does.** It applies rounds of $m$ disjoint random
pairs. Within a round the pairs are disjoint, so their costs are evaluated on
the same ranking and add, and each pair is marginally a uniform random pair,
so by linearity a round multiplies $\mathbb{E}[C]$ by exactly
$1 - 2m/(n-1)$. Rounds of $m = \lfloor n/4 \rfloor$ roughly halve the IC;
the last round takes whatever fraction remains, $m_{\text{last}} =
(1 - \phi)(n-1)/2$, which is not an integer. Because the factor is linear in
$m$, rounding $m_{\text{last}}$ up with probability equal to its fractional
part makes the expectation exact rather than off by part of a swap. No swap
cost is ever evaluated, which is why this is the fastest mode.

Random swaps only ever shrink $\mathbb{E}[C]$ towards zero, so `random`
requires $\rho^\star > 0$; negative targets are what `tails` and `middle`
(full reversal gives $-\rho_{\max}$) are for.

## 4. Per-decision scatter

The swap count in §3 is fixed by the formula and no decision is steered to
the target, so each decision's IC is a random variable with mean
$\rho^\star$. What is its spread?

For a small target the final ranking is close to a uniformly random
permutation, and Spearman's $\rho$ of a random permutation has mean $0$ and
**variance exactly $1/(n-1)$**, with a distribution that is Gaussian to a
very good approximation once $n$ is a few dozen (the exact permutation
distribution is slightly platykurtic, with excess kurtosis of order $-1/n$).
Shifting by the mean,

$$
\rho_d \;\sim\; \mathcal N\!\left(\rho^\star,\ \frac{1}{n-1}\right)
\quad\text{approximately, per decision.}
$$

This is not an arbitrary amount of noise. A *real* forecast with true IC
$\rho^\star$ scored on $n$ names shows exactly this sampling scatter per
decision, because its per-decision IC is a correlation estimated from $n$
pairs. So `random` reproduces both the mean and the decision-to-decision
variability of a genuine forecast of that quality, for free. (At large
targets, say $\rho^\star = 0.5$, the walk has not fully mixed and the
variance is a few percent above $1/(n-1)$.)

**Giving `tails` and `middle` the same scatter.** `scatter="sampling"` draws
each decision's target IC from $\mathcal N(\rho^\star, 1/(n-1))$, clipped to
$[-\rho_{\max}, \rho_{\max}]$, and then hits it exactly with the §2 budget.
The three modes then share a per-decision IC distribution and differ only in
*where* the errors sit.

Why sample the IC rather than the swap count? Because the count is far too
coarse a variable. From §2, one outer pair moves the IC by about $12/n$ and
one middle pair near the target by about $7.6/n$; an IC spread of
$1/\sqrt{n-1}$ is therefore a count spread of only $\sqrt{n}/12$ to
$\sqrt n/7.6$ pairs, i.e. two or three pairs for a few hundred names. A
Gaussian on an integer with that standard deviation gives a lumpy IC
distribution, and with ties the count-to-IC map is not even fixed. Sampling
the IC and letting the exact budget find the swaps gives the correct induced
distribution on counts automatically. Gaussian is the right family here, not
a placeholder: it is what the random-permutation distribution converges to.

## 5. Forecast values and scale

The forecast's *values* are the realized returns of the cross-section,
reassigned according to the degraded ranking, so the forecast is in return
units and its ranking is the only thing that was changed. A tiny strictly
increasing offset is added so the forecast carries no ties of its own, since
a tie would average two forecast ranks and move the IC off target.

Those values are then shrunk towards the cross-sectional mean by `scale`,
defaulting to $|\rho^\star|$. An unshrunk oracle has the full dispersion of
realized returns, which no forecast with an IC of $0.02$ could claim; a
cost-aware allocator or `edge_to_cost` would read it as an edge $1/\rho^\star$
times too large. The default is Grinold's
$\alpha = \mathrm{IC} \times \sigma \times \text{score}$: the forecast size a
forecaster who believed its own IC would produce. Any positive scale leaves
the ranking, and hence the IC, unchanged.

## 6. Reference

| Quantity | Formula |
|---|---|
| IC | $\rho = C / \sqrt{S_f S_r}$, $C = \sum (f_i - \bar f)(r_i - \bar r)$ |
| Cost of swapping forecast ranks of $a, b$ | $\Delta C = -(f_a - f_b)(r_a - r_b)$ |
| Mirror pair $i$, no ties | $(n-1-2i)^2$; outermost pair $\approx 12/n$ in IC |
| `middle`, $k$ pairs, no ties, even $n$ | $\rho \approx 1 - 16k^3/n^3$, so $k \approx n((1-\rho)/16)^{1/3}$ |
| Expected cost of a random swap | $2C/(n-1)$, any ranking |
| Walk with replacement | $\mathbb E[\rho_k] = \rho_{\max}(1 - 2/(n-1))^k$; $k \approx \tfrac{n-1}{2}\ln(\rho_{\max}/\rho^\star)$ |
| Disjoint pairs | $\mathbb E[\rho_k] = 1 - 2k/(n-1)$; $k = (1-\rho^\star)(n-1)/2$ |
| Round of $m$ disjoint random pairs | multiplies $\mathbb E[C]$ by $1 - 2m/(n-1)$ exactly |
| Per-decision scatter | $\mathrm{Var}(\rho_d) \approx 1/(n-1)$, Gaussian |
| Default value scale | $|\rho^\star|$ (Grinold) |

## 7. Empirical checks

Synthetic $t_3$ returns, 4,000 decisions per row, `random` mode. The $n$
values are just the two sizes tried; the formulas take $n$ from the data.

| $n$ | target | mean IC (s.e.) | per-decision sd | $1/\sqrt{n-1}$ |
|---|---|---|---|---|
| 700 | 0.01 | 0.0100 (0.0006) | 0.037 | 0.038 |
| 700 | 0.02 | 0.0192 (0.0006) | 0.039 | 0.038 |
| 700 | 0.05 | 0.0503 (0.0006) | 0.038 | 0.038 |
| 700, heavy ties | 0.02 | 0.0189 (0.0006) | 0.039 | 0.038 |
| 50 | 0.02 | 0.0166 (0.0023) | 0.143 | 0.143 |

For `tails` and `middle` without scatter, every decision's IC is within
$3\times10^{-5}$ of the target on the same data, and within $5\times10^{-5}$
with heavy ties.
