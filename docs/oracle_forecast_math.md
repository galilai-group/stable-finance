# The mathematics of `oracle_forecast`, `copula_forecast` and `factor_forecast`

`oracle_forecast` turns realized returns into a forecast whose rank IC is known
in advance, so that portfolio construction and trade calculation can be tested
against a forecast with no unknowns of its own. This note derives everything
the implementation relies on (§§1–6), and then the separate generator
`copula_forecast`, whose errors belong to the names rather than to rank
positions (§7), and `factor_forecast`, the copula with factor-structured
errors, a time-varying IC and name-dependent accuracy (§8). Throughout, $n$
is the number of names with a
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
(`copula_forecast` has no exact target to protect and adds no offset; see
§7.)

Those values are then shrunk towards the cross-sectional mean by `scale`,
defaulting to $|\rho^\star|$. An unshrunk oracle has the full dispersion of
realized returns, which no forecast with an IC of $0.02$ could claim; a
cost-aware allocator or `edge_to_cost` would read it as an edge $1/\rho^\star$
times too large. The default is Grinold's
$\alpha = \mathrm{IC} \times \sigma \times \text{score}$: the forecast size a
forecaster who believed its own IC would produce. Any positive scale leaves
the ranking, and hence the IC, unchanged.

## 6. Per-horizon targets and persistent errors

**Per-horizon IC.** A real forecast's IC decays with horizon, so `ic` may be a
vector $\rho^\star_h$ with one target per horizon. Nothing in §§1–5 couples
horizons: each `(decision, horizon)` cross-section is degraded with its own
$\rho^\star_h$ and, by default, scale $|\rho^\star_h|$. The random draws come
from one stream consumed decision-major, horizon-minor, exactly as for a
scalar target, so a constant vector reproduces the scalar result bit for bit.

**Persistence.** By default the degradation of decision $d$ is a fresh
random permutation $\pi_d$ of rank positions, independent across decisions.
With `persistence=k` decisions are grouped into blocks -- $b = \lfloor d/k
\rfloor$, or, given `days`, runs of at most $k$ decisions that restart at
every change of day label -- and every random draw for decision $d$ at
horizon $h$ comes from a generator seeded with $(\text{root}, h, b)$, i.e.
`np.random.default_rng([root, h, b])`, where `root` is the integer `seed`, or
one integer drawn from `np.random.default_rng(seed)` when `seed` is a
generator or `None`. Every decision of a block replays the same draws, so:

* under `random`, $\pi_d = \pi_b$ for every $d$ in the block -- the map from
  realized rank to forecast rank is identical -- provided the decisions have
  the same $n$ and no ties (with ties the factor $\rho^\star \sqrt{S_fS_r} /
  C_{\max}$ and hence the round sizes of §3 vary with the decision);
* under `tails` and `middle` the mirror pairs are already a fixed function
  of rank position, so only the `scatter` draw is shared: one standardized
  Gaussian $z_b$ per block, target $\rho^\star + z_b/\sqrt{n-1}$.

`persistence=None` and `persistence=1` keep the single shared stream and are
the default behaviour exactly.

*What is guaranteed.* Marginally nothing changes: $\pi_b$ has the law $\pi_d$
had, so each decision's IC has the mean and $1/(n-1)$ variance of §4. What
changes is the joint law across decisions. Without ties the IC of a rank
permutation does not depend on the values permuted, so under `random` the IC
is *constant* within a block, and the block mean of the IC has variance
$1/(n-1)$ rather than $1/(k(n-1))$: the decision-to-decision scatter no
longer averages away. Measured by rank position, the error $e_d(j) =
\pi_d(j) - j$ is identical within a block (correlation $1$), against about
$(1-\rho^\star)/2$ between independent draws, which share only their common
pull towards the middle, $\mathbb E[\pi(j) - j] \approx (1-\rho^\star)(\bar
f - j)$.

*What is not guaranteed.* The errors persist in rank space, not by name. A
name $i$ with realized rank $r_{d,i}$ has forecast rank $\pi_b(r_{d,i})$, and
under `random` $\pi_b$ is far from smooth: two adjacent rank positions are
sent to unrelated places. The forecast is therefore autocorrelated across
decisions only in so far as names keep exactly their realized rank. On 200
names with `ic=0.05` and `persistence=10`, the forecast's lag-one rank
autocorrelation is

| realized rank autocorrelation | `persistence=None` | `persistence=10` |
|---|---|---|
| 0.945 | 0.01 | 0.04 |
| 0.988 | 0.01 | 0.08 |
| 0.998 | 0.00 | 0.20 |
| 1 (frozen within block) | 0.01 | 0.91 |

`tails` and `middle` move names in contiguous runs of rank positions, so
their forecasts already inherit most of the truth's persistence without
`persistence` (0.6–0.9 in the same experiment) and gain a few hundredths
from it. Persistence is thus a model of a forecast whose *rank-position
errors* and *IC* are sticky; on its own it does not turn a fast `random`
oracle into a slowly-moving forecast, and a study of turnover should check
the forecast's own autocorrelation rather than assume it. A forecast whose
errors persist by name is what `copula_forecast` (§7) generates.

## 7. A separate generator: `copula_forecast`

Everything above degrades a ranking by exchanging rank positions, so its
errors belong to rank positions. §6 showed what that means for persistence:
the object that repeats is a permutation $\pi_b$ of positions, and a name
keeps its error only while it keeps its realized rank. A real forecast's
errors belong to the *name*: a stale view of a stock stays wrong about that
stock however the market reorders around it. `copula_forecast` generates
such errors with a different model and different parameters -- a half-life
rather than a block length, an IC hit in expectation rather than steered --
which is why it is a function of its own rather than a fourth `noise`.

**The model.** In each cross-section let $z_i = \Phi^{-1}(r_i/(n+1))$ be the
normal score of name $i$'s realized average rank. The forecast score is

$$
s_i = c\, z_i + \sqrt{1-c^2}\; e_i ,
$$

with $e_i$ standard-normal noise owned by name $i$, and the forecast is the
ranking of $s$. A bivariate normal with Pearson correlation $c$ has
Spearman correlation $\tfrac{6}{\pi}\arcsin(c/2)$, so

$$
c = 2 \sin\!\left(\frac{\pi \rho^\star}{6}\right)
$$

gives rank IC $\rho^\star$ in expectation. $z$ is a fixed grid of normal
quantiles rather than a Gaussian sample, and a sample Spearman correlation is
biased by $O(1/n)$, so the expectation falls short of $\rho^\star$ by a
relative amount of order $1/n$: measured, about 8% at $n = 30$, 2% at
$n = 120$ and 0.5% at $n = 400$. No decision is steered, so each decision's
IC scatters with standard deviation about $(1 - \rho^{\star 2})/\sqrt{n-1}$,
which at the ICs real forecasts have is the $1/\sqrt{n-1}$ of §4. $\rho^\star$
may be negative; $|\rho^\star| = 1$ is excluded, since $c = \pm 1$ leaves
no noise at all (`oracle_forecast(noise="tails")` gives the perfect and the
fully reversed ranking).

**Half-life.** With `half_life=k` each name's noise is an AR(1) across
decisions,

$$
e_{d,i} = \phi\, e_{d-1,i} + \sqrt{1-\phi^2}\; \eta_{d,i},
\qquad \phi = 2^{-1/k},
\qquad \eta_{d,i} \sim \mathcal N(0, 1) \text{ i.i.d.}
$$

It is stationary with unit variance, so a single decision's IC has the law
above whatever $k$ is. The correlation of a name's noise $j$ decisions apart
is $\phi^j = 2^{-j/k}$, which halves every $k$ decisions. On realized
returns independent across decisions, $\operatorname{Corr}(s_{d,i},
s_{d+1,i}) = (1-c^2)\phi$, and the forecast's lag-one rank
autocorrelation is close to it (120 names, 1,000 decisions):

| `half_life` | `ic` | $(1-c^2)\phi$ | measured |
|---|---|---|---|
| `None` | 0.05 | 0 | 0.01 |
| 12 | 0.05 | 0.941 | 0.934 |
| 12 | 0.3 | 0.851 | 0.843 |

against 0.08 for rank-space persistence on realized returns whose own rank
autocorrelation is 0.99 (§6). When the realized ranking itself persists,
the $c z$ term adds to this. `half_life=None` sets $\phi = 0$: fresh
noise at every decision. `half_life=math.inf` sets $\phi = 1$ and freezes
each name's noise. Given `days`, every name's noise restarts from a fresh
$\mathcal N(0,1)$ draw at each change of day label, so no error is carried
over a night. A name whose realized return is missing gets no forecast, but
its noise keeps evolving, so the error is still its own when it returns.

**Values and ties.** The realized values of the cross-section, sorted, are
handed out in score order -- the name with the $i$-th smallest score gets
the $i$-th smallest realized return -- and then shrunk towards the mean by
`scale` as in §5. No offset is added: where the realized returns are tied,
the forecast values at the corresponding score ranks are tied too, and the
rank IC scores them with average ranks exactly as it scores the tied
outcomes. §5's offset exists to keep `tails` and `middle` on an exact target;
the copula has none to protect.

**Randomness.** Horizon $h$ draws all of its noise from
`np.random.default_rng([root, h])`, with `root` as in §6: the stationary
start, then one innovation per decision. Errors are independent across
horizons, and a horizon's forecast does not depend on how many horizons
there are.

## 8. A third generator: `factor_forecast`

`copula_forecast` gives each name its own error, but three things about it
are still unlike a real forecast, and each biases a study in a known
direction. Its errors are independent *across names*, so they diversify away
over a few hundred names, breadth looks larger than it is, and neutralizing
a factor or using a better risk model can only remove true signal. Its IC is
the same at every decision, so the only strategy risk left is the
$1/\sqrt{n-1}$ sampling scatter of §4, while a real forecast's realized IR
is $\overline{\mathrm{IC}}/\mathrm{sd}(\mathrm{IC}_t)$ (Qian and Hua, 2004;
Ding and Martin, 2017) and is set by its good and bad stretches. And every
name is forecast equally well, while short-horizon predictability
concentrates in wide-spread, volatile names. `factor_forecast` adds the
three to the copula, each by a parameter; with `factor_share=0`,
`ic_dispersion=0` and `quality=None` it is `copula_forecast`, draw for draw.

**The model.** With $z_{t,i}$ the normal score of name $i$'s realized rank
at decision $t$, as in §7,

$$
s_{t,i} = c_{t,i}\, z_{t,i} + \sqrt{1-c_{t,i}^2}\; e_{t,i},
\qquad
e_{t,i} = g_{t,i} + \sqrt{1-\phi_i}\; u_{t,i},
$$

where $u$ is the copula's noise (AR(1) by name with the half-life, drawn
from the copula's own stream), $g_t$ is the factor part of the error and
$\phi_i$ is name $i$'s factor share. The forecast is the ranking of $s$,
valued as in §7.

**Loadings.** Let $X$ be the history's first horizon with each cross-section
replaced by its normal scores (0 where a name is missing), and each column
(name) centred over time. With $X = U \Sigma V^\top$, the loadings are
$B = V_{:,1:k}\,\Sigma_{1:k}$, scaled so that $\overline{|B_i|^2} = 1$ over
the names that have any, with each column's sign fixed so that its largest
entry is positive. Normal scores rather than returns because the score lives
in normal-score space, because a shift common to every name changes no
ranking -- each row of $X$ sums to zero, so $\mathbf 1$ is in the null space
of $X^\top X$ and every loading vector is orthogonal to it: the market's
level is not a factor -- and because no single outlier or split can capture
a component. With no history the loadings come from `realized` itself.

**Shares.** Name $i$'s share is proportional to its squared loading, capped
at one, and scaled so that the shares average $\phi$ = `factor_share`:

$$
\phi_i = \min(\lambda |B_i|^2,\ 1),
\qquad
\frac1N\sum_i \phi_i = \phi ,
$$

solved for $\lambda$ by water-filling (cap the largest names one at a time
until the largest uncapped one fits). A name that moves with the factors is
wrong with them; one the factors barely touch keeps a mostly idiosyncratic
error. On August 2020 with three factors, $|B_i|^2$ runs from 0.01 to 5.8
times its mean (median 0.68), so at $\phi = 0.3$ 4.4% of names are capped,
and 29% at $\phi = 0.6$. At $\phi = 1$ every name with a loading has share
one.

**The factor part.** With $f_t \sim \mathcal N(0, I_k)$ (AR(1) with the
half-life, restarting each day) and $\hat b_i = B_i/|B_i|$,

$$
G_{t,i} = \sqrt{\phi_i}\; \hat b_i^\top f_t ,
\qquad
g_{t,i} = \bigl(G_{t,i} - \bar G_t\bigr)\,
          \frac{\sqrt{\overline{\phi}_t}}{\operatorname{sd}_i(G_{t,i})},
$$

where the mean, standard deviation and $\overline{\phi}_t$ are over the
names of the cross-section. $G_{t,i}$ has variance $\phi_i$ over time, so
each name's error has unit variance, and

$$
\operatorname{Corr}(e_{t,i}, e_{t,j}) \approx \sqrt{\phi_i \phi_j}\;
\hat b_i^\top \hat b_j .
$$

The cross-sectional rescaling makes the factors carry exactly
$\overline\phi_t \approx \phi$ of the error's cross-sectional variance at
every decision (up to the sample variance of $u$, the copula's). Without it,
$\operatorname{var}_i(G_{t,\cdot}) = f_t^\top \Omega f_t$ with $\Omega$ the
cross-sectional covariance of the exposures would swing with $f_t$ -- with
one dominant factor it is nearly a $\chi^2_1$ -- and the IC, a concave
function of the error variance, would move off the copula's: about
$+\tfrac38\phi^2\operatorname{Var}(f^\top\Omega f)/(\mathbb E f^\top\Omega f)^2$
relatively, 3% at $\phi = 0.3$ and 9% at $\phi = 0.5$ with August 2020's
exposures, whose $\Omega$ has eigenvalue shares 0.61, 0.27 and 0.12.

**The per-decision IC.** Conditional on the decision, let
$\rho_t = \operatorname{corr}_i(e_{t,i}, z_{t,i})$, so that
$\rho_t \approx \sqrt\phi\, r_t + \sqrt{1-\phi}\, r^u_t$ with
$r_t = \operatorname{corr}_i(g_t, z_t)$ and
$r^u_t \sim \mathcal N(0, 1/(n-1))$. With unit variances the score's
correlation with $z$ is

$$
\operatorname{corr}_i(s, z) = \frac{c + a\rho_t}{\sqrt{1 + 2ac\rho_t}},
\qquad a = \sqrt{1-c^2}.
$$

Its slope at $\rho = 0$ is $a^3$, so the IC's spread is

$$
\operatorname{sd}(\mathrm{IC}_t) \approx \sqrt{1-c^2}\,
\sqrt{\phi\, \mathbb E[r_t^2] + (1-\phi)/(n-1)} .
$$

When the realized returns share the factor structure, $r_t$ is far larger
than $1/\sqrt n$: an error that bets on a factor is right or wrong about
many names at once. On August 2020 the 5-minute cross-section's absolute
correlation with the first three loading vectors averages 0.26, 0.17 and
0.13, and at $\phi = 0.3$ $\mathbb E[r_t^2] = 0.068$ (0.101 with one
factor, 0.049 with ten), so the formula gives
$\operatorname{sd}(\mathrm{IC}_t) \approx 0.146$, against the copula's
0.039 (measured in §10: 0.151). Real forecasts show 0.12 (ridge, LASSO) to
0.15 (15-minute reversal) on the same month, so `factor_share` between 0.2
and 0.3 reproduces them. On returns without factor structure,
$\mathbb E[r_t^2] \approx 1/(n-1)$ and the spread is the copula's.

Its second derivative at $\rho = 0$ is $-a^2 c\,(2 - 3c^2)$, so

$$
\mathbb E[\operatorname{corr}_i(s, z)] \approx
c\,\bigl(1 - a^2 (1 - \tfrac32 c^2)\, \mathbb E[\rho_t^2]\bigr).
$$

This is the mechanism of the copula's own $O(1/n)$ shortfall (§7), with
$\mathbb E[\rho_t^2] = 1/(n-1)$; factor errors grow it, and the mean IC
falls short of the copula's by a relative amount of about
$\phi\,(\mathbb E[r_t^2] - 1/(n-1))$: 2.0% at $\phi = 0.3$ and 3.8% at
$\phi = 0.6$ on August 2020 by the formula, 2.5% and 1.8% measured over four
seeds (each $\pm 2\%$, §10). That is inside one month's standard error of
the mean IC, which the wider spread has itself grown to about
$0.15/\sqrt{1638} \approx 0.0037$. It is left in
rather than calibrated away: it is a property of a forecast whose errors are
bets, and every scored row reports the IC it actually had. On factor-free
returns the mean matches the copula's within about 1% at $\phi = 0.3$, and
within a few percent at $\phi = 1$, where the error is a $k$-dimensional
pattern whose distribution across names is far from Gaussian and the
Pearson-to-Spearman map of §7 no longer holds exactly.

**Time-varying IC.** With `ic_dispersion` $= \delta > 0$ the target at
decision $t$ is

$$
\kappa_t = \rho^\star + \delta\, a_t,
\qquad a_t = \psi\, a_{t-1} + \sqrt{1-\psi^2}\;\eta_t,
\qquad a_0, \eta_t \sim \mathcal N(0,1),
$$

with $\psi$ = `ic_persistence` (0 when `None`), restarting from
$\mathcal N(0, 1)$ at each new day when `days` is given, and
$c_t = 2\sin(\pi\kappa_t/6)$ clipped to $|\kappa_t| \le 0.999$. The
expected IC is then an AR(1) with mean $\rho^\star$, standard deviation
$\delta$ and autocorrelation $\psi^j$, and the measured IC adds the
sampling (and factor) scatter $\epsilon_t$, white across decisions when the
errors are drawn afresh:

$$
\mathbb E[\mathrm{IC}_t] \approx \rho^\star, \qquad
\operatorname{Cov}(\mathrm{IC}_t, \mathrm{IC}_{t+j}) = \delta^2\psi^j
\ (j \ge 1), \qquad
\operatorname{Var}(\mathrm{IC}_t) = \delta^2 + \operatorname{Var}(\epsilon_t).
$$

The measured IC is therefore an ARMA(1,1), not an AR(1): its lag-one
autocorrelation is $\psi\,\delta^2/(\delta^2 + \operatorname{Var}\epsilon)$,
much less than $\psi$ when the scatter dominates. $\psi = 1$ draws one
target per day (with `days`), a pure day-level regime.

**Quality.** With `quality` set, name $i$'s target is $\kappa_t q_{t,i}$,
with $q$ linear in the name's rank $R$ by the quality variable among the
$m$ names of the cross-section that have one,

$$
q = 1 + \lambda_q\Bigl(\frac{2(R - \tfrac12)}{m} - 1\Bigr),
$$

$\lambda_q$ = `quality_strength`, so that at $\lambda_q = 1$ the weights
run from about 0 to about 2 (a name without the variable gets 1). The
weights are then rescaled per decision to keep the rank IC. To first order
in $c$, with $u_i = \Phi(z_i) = R^{\text{real}}_i/(n+1)$,

$$
\rho_S = 12\,\mathbb E[F_z(z) F_s(s)] - 3
\approx \frac{6}{\sqrt\pi}\,\frac1n \sum_i c_i\, z_i \bigl(u_i - \tfrac12\bigr),
$$

since $\mathbb E[F_s(s_i)\mid z_i] \approx \tfrac12 + c_i z_i\,
\mathbb E[\varphi(e)]$ with $\mathbb E[\varphi(e)] = 1/(2\sqrt\pi)$ for
standard-normal $e$. (For constant $c$,
$\tfrac1n\sum z(\Phi(z) - \tfrac12) \to 1/(2\sqrt\pi)$ and this is the
$3c/\pi$ of $\tfrac6\pi\arcsin(c/2)$.) The weights are therefore rescaled so
that $\sum_i q_i\, z_i (u_i - \tfrac12) = \sum_i z_i (u_i - \tfrac12)$.
Weights that merely average one are not enough when the variable tracks
$|z|$, as volatility does: on synthetic returns with log-normal
volatilities (sd 0.5 in logs) across names and a target of 0.2, unrescaled
weights raised the IC by about 28%, weights rescaled to keep
$\sum q z^2$ lowered it by about 4%, and the leverage
$z(\Phi(z) - \tfrac12)$ kept it within about 1%. The
noise's size is never varied on its own: every score keeps unit variance, so
the tails of the forecast are not filled with the noisiest names.
`"spread"` reads each decision's half-spread from `market`; `"vol"` uses
each name's standard deviation over the history's first horizon. Even the
copula forecasts volatile names' *ranks* better on heteroskedastic returns,
since their realized ranks sit in the tails where a given $c$ places them
most reliably (0.27 in the most volatile third against 0.13 in the calmest
at IC 0.2 in the same synthetic data); `"vol"` adds to that.

**Persistence and randomness.** `half_life` $k$ makes both $u$ (as in §7)
and $f_t$ AR(1) with $\phi = 2^{-1/k}$. The rescaled $g_t$ follows $f_t$'s
direction, whose lag correlation is a little below $\phi$ -- with one
factor, $g_t \propto \operatorname{sign}(f_t)$ and the correlation is
$\tfrac2\pi\arcsin\phi$. Horizon $h$ draws $u$ from
`default_rng([root, h])` exactly as `copula_forecast` does, $f_t$ from
`default_rng([root, h, 1])` and $a_t$ from `default_rng([root, h, 2])`.
Note that numpy's `SeedSequence` ignores trailing zeros, so
`default_rng([root, 0])` is `default_rng(root)`: a test panel generated
from `default_rng(seed)` with the same integer as the forecast's seed
shares horizon 0's noise stream (this holds for `copula_forecast` too).

## 9. Reference

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
| Default value scale | $|\rho^\star|$ (Grinold), per horizon |
| Persistent block generator | `default_rng([root, h, b])`; marginal IC law unchanged |
| `copula_forecast` score | $s = c z + \sqrt{1-c^2}\,e$, $c = 2\sin(\pi\rho^\star/6)$ |
| `copula_forecast` noise | AR(1), $\phi = 2^{-1/\text{half\_life}}$; restarts at each new day |
| `copula_forecast` lag-one autocorrelation, i.i.d. returns | $\approx (1-c^2)\phi$ |
| `copula_forecast` generator | `default_rng([root, h])` |
| `factor_forecast` score | $s = c_{t,i} z + \sqrt{1-c_{t,i}^2}\,(g_t + \sqrt{1-\phi_i}\,u)$ |
| Loadings | top $k$ of SVD of name-centred normal scores, $B = V_k\Sigma_k$, $\overline{\lvert B_i\rvert^2} = 1$ |
| Factor shares | $\phi_i = \min(\lambda\lvert B_i\rvert^2, 1)$, $\bar\phi = $ `factor_share` |
| Factor part | $\sqrt{\phi_i}\,\hat b_i^\top f_t$, rescaled across names to variance $\overline{\phi}_t$ |
| Error correlation | $\approx \sqrt{\phi_i\phi_j}\,\hat b_i^\top \hat b_j$ |
| IC spread | $\approx \sqrt{1-c^2}\,\sqrt{\phi\,\mathbb E r_t^2 + (1-\phi)/(n-1)}$ |
| Mean-IC shortfall vs the copula | $\approx \phi\,(\mathbb E r_t^2 - 1/(n-1))$, relative |
| Time-varying target | $\kappa_t = \rho^\star + \delta a_t$, $a_t$ AR(1) with $\psi$; $\operatorname{Cov}(\mathrm{IC}_t, \mathrm{IC}_{t+j}) = \delta^2\psi^j$ |
| Quality weights | $q = 1 + \lambda_q(2(R-\tfrac12)/m - 1)$, rescaled to keep $\sum q\, z(\Phi(z)-\tfrac12)$ |
| `factor_forecast` generators | $u$: `default_rng([root, h])`; $f$: `[root, h, 1]`; $a$: `[root, h, 2]` |

## 10. Empirical checks

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

**`factor_forecast`, synthetic.** Time-varying IC, i.i.d. Gaussian returns,
3,000 decisions, $n = 300$, target 0.05, `factor_share=0`; predictions from
§8, with sampling variance $(1-0.05^2)^2/299$:

| $\delta$ | $\psi$ | Var(IC) | pred. | lag 1 | pred. | lag 2 | pred. | lag 5 | pred. |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.8 | 0.00575 | 0.00583 | 0.00201 | 0.00200 | 0.00154 | 0.00160 | 0.00080 | 0.00082 |
| 0.05 | `None` | 0.00576 | 0.00583 | 0.00002 | 0 | -0.00004 | 0 | 0.00003 | 0 |
| 0.03 | 0.95 | 0.00415 | 0.00423 | 0.00082 | 0.00085 | 0.00074 | 0.00081 | 0.00065 | 0.00070 |

With $\psi = 1$, `days` of 100 decisions and $\delta = 0.05$, the day means
of the IC spread by 0.056 (predicted $\sqrt{\delta^2 + 0.0033/100} =
0.050$) and the IC within a day by 0.057 (the sampling 0.058).

Error structure on i.i.d. returns (400 decisions, $n = 120$) with loadings
from a three-factor history: the share of each decision's error, with its
part along $z$ removed, that the exposures $\sqrt{\phi_i}\hat b_i$ explain
across names is 0.025 / 0.50 / 0.96 at $\phi$ = 0 / 0.5 / 1 (the
regression's floor is $k/n = 0.025$), and the slope of the errors' pairwise
time-series correlation on $\sqrt{\phi_i\phi_j}\,\hat b_i^\top\hat b_j$
is 0.00 / 0.97 / 0.95.

Quality on i.i.d. homoskedastic returns (600 decisions, $n = 300$, target
0.2, `factor_share=0.3`), IC within the names' thirds by half-spread
against the weights' averages $0.2(1 \mp \tfrac23\lambda_q)$:

| $\lambda_q$ | narrowest third | pred. | widest third | pred. | all |
|---|---|---|---|---|---|
| 1 | 0.077 | 0.067 | 0.327 | 0.333 | 0.198 |
| 0.5 | 0.143 | 0.133 | 0.262 | 0.267 | 0.197 |
| 0 | 0.207 | 0.200 | 0.197 | 0.200 | 0.197 |

**`factor_forecast` on the real month.** August 2020 (1,638 decisions, 666
names, 21 days), loadings and volatilities from July 2020's 5-minute
returns, three factors, `days` given, four seeds (two for the last three
rows), 5-minute horizon unless stated. Lag correlations are within days.
"Explained" is the cross-sectional $R^2$ of the error (part along $z$
removed) on the exposures, "slope" that of the pairwise error correlation on
its prediction, both from seed 0; on real returns, which themselves load on
the factors, removing the part along $z$ removes some factor error too,
which is why they fall short of $\phi$ and 1.

| forecast | target | mean IC (seed s.e.) | sd(IC$_t$) | lag 1 | lag 3 | sd of day means | explained | slope |
|---|---|---|---|---|---|---|---|---|
| copula | 0.03 | 0.0310 (0.0004) | 0.039 | 0.00 | -0.01 | 0.005 | 0.004 | 0.00 |
| $\phi = 0.3$ | 0.03 | 0.0304 (0.0021) | 0.151 | 0.00 | 0.01 | 0.017 | 0.267 | 0.80 |
| $\phi = 0.6$ | 0.03 | 0.0315 (0.0028) | 0.209 | 0.00 | 0.01 | 0.022 | 0.522 | 0.82 |
| copula | 0.1 | 0.1005 (0.0003) | 0.039 | 0.00 | -0.01 | 0.005 | | |
| $\phi = 0.3$ | 0.1 | 0.0979 (0.0020) | 0.149 | 0.00 | 0.01 | 0.017 | | |
| $\phi = 0.6$ | 0.1 | 0.0988 (0.0027) | 0.206 | 0.00 | 0.01 | 0.022 | | |
| $\phi = 0.3$, half-life 12 | 0.03 | 0.0276 (0.0012) | 0.151 | 0.01 | 0.00 | 0.015 | | |
| same, 900 s | 0.03 | 0.0299 (0.0035) | 0.156 | 0.62 | 0.02 | 0.030 | | |
| same, 1800 s | 0.03 | 0.0321 (0.0049) | 0.154 | 0.76 | 0.37 | 0.035 | | |
| $\phi = 0.3$, $\delta = 0.02$, $\psi = 0.9$ | 0.03 | 0.0313 (0.0032) | 0.153 | 0.01 | 0.02 | 0.019 | | |

Averaged over the three horizons (300, 900, 1800 s) at target 0.1 the mean
IC is 0.0998 for the copula, 0.0973 at $\phi = 0.3$ and 0.0980 at
$\phi = 0.6$. For comparison, the real signals on the same month: per-decision
sd(IC) 0.147 / 0.117 / 0.120 (15-minute reversal / ridge / LASSO) at 300 s,
lag-one IC correlation 0.06 at 300 s and 0.14 to 0.55 at 900 and 1800 s,
where overlapping horizons correlate neighbouring decisions' outcomes, and
sd of day means 0.017 to 0.034 -- where the copula's 0.005 shows how little
strategy risk independent errors leave. The predicted lag-one correlation
of the $\delta = 0.02$, $\psi = 0.9$ row is
$\psi\delta^2/\operatorname{Var}(\mathrm{IC}_t) = 0.9 \cdot 0.02^2/0.153^2
\approx 0.015$; measured 0.013.

Quality at $\phi = 0.3$, target 0.03, 5-minute horizon, IC within the thirds
of names by each decision's half-spread (or by July volatility): `"spread"`
0.013 in the narrowest third and 0.057 in the widest, against 0.031 and
0.039 without quality; `"vol"` 0.009 and 0.062, against 0.026 and 0.044.
The whole cross-section's IC is unchanged: at target 0.1 and three seeds it
is 0.1008 / 0.1007 / 0.1006 without quality, with `"spread"` and with
`"vol"` at $\phi = 0$, and 0.0990 / 0.0987 / 0.0983 at $\phi = 0.3$.

Generating all three horizons takes 3.2 s (5.5 s with quality) on one core.
