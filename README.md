# Limit-cycle neural surrogates

Why learned surrogates of oscillatory systems lose the oscillation, and what to
constrain instead.

Short answer: the error that matters is phase, not amplitude. It is set by a
single scalar — the relative period error δ — which is invisible in the training
loss and untouched by every stability method currently in use. Carrying phase as
an explicit coordinate with a measured frequency makes the rollout period exact
by construction.

Where it bites is **parametric** models. Fitting one operating point, a free
surrogate reaches δ ≈ 2 × 10⁻⁵ and holds a cylinder wake for ~10⁴ periods — phase
drift is not what limits it. Condition the same architecture on a parameter and
δ inflates by two orders of magnitude, on an ODE and on Navier–Stokes alike:
~500 periods to decorrelation at held-out Reynolds numbers, 14 when
extrapolating.

Anchoring the phase to a measured frequency recovers most of that
in-distribution — 13× at training parameters — but **not** outside the training
range, where it is worse than doing nothing. The guarantee covers the phase
variable, not the decoded observable, and the difference is demonstrated rather
than assumed.

Two results that cut against the obvious story are reported anyway: the
single-point wake above, and the fact that the first anchored parametrization
tried here made δ *worse*, for a reason worth knowing.

Every number below comes from the scripts in this repository.

## The argument

A stable limit cycle has one Floquet multiplier exactly equal to 1, along the
orbit, forced by time-translation invariance. The rest lie inside the unit
circle. So a surrogate's amplitude error decays on its own, and its phase error
does not. A relative period error δ produces

    Δθ(t) = (Ω̂ − Ω) t ≈ −2π δ t / T

which grows without bound until the prediction is antiphase with the truth and
relative L² saturates near √2. That saturating curve is what usually gets
plotted and called long-horizon drift.

## Results

Stuart–Landau oscillator (μ=0.5, ω=1, β=−0.5), where the cycle radius √μ,
frequency ω−βμ and transverse multiplier e^(−2μT) are known in closed form.
Neural ODE with a tanh MLP vector field, RK4 rollout training, 3000 epochs,
five seeds per configuration, 200-period evaluation rollouts.

### The phase law is exact

Over 70 trained models, the observed phase-drift rate divided by the rate
predicted from the measured period error:

| quantity | value |
|---|---|
| median observed/predicted slope | 1.0000 (range 1.000–1.033) |
| residual from a straight line fit to Δθ(t) | 1.5 × 10⁻³ rad |
| max deviation of computed \|λ₁\| from 1 | 1.2 × 10⁻¹⁰ |

Phase error is fully accounted for by the period error. Nothing else is needed
to explain it.

### Phase is unbounded, amplitude is not

| | median | worst |
|---|---|---|
| \|Δθ\| at 200 periods | 0.82 rad | 52.0 rad |
| max amplitude error | 3.6 × 10⁻³ | 0.12 |

Cycle radius is 0.707 for scale.

### Training loss does not predict long-horizon fidelity

Spearman correlation between log(train MSE) and log\|δ\| over 35 runs spanning a
64× range of dataset size: **ρ = −0.22, p = 0.20**. Among the three lowest-loss
models δ still spans a factor of 3. Two runs with the same architecture, same
data, indistinguishable loss:

| trajectories | width | seed | train MSE | δ | decorrelation |
|---|---|---|---|---|---|
| 64 | 32 | 3 | 9.1 × 10⁻⁶ | +1.4 × 10⁻³ | 179 T |
| 64 | 32 | 4 | 7.2 × 10⁻⁶ | +1.1 × 10⁻⁵ | > 200 T |

### Scaling does not fix it

Median \|δ\| by training set size, at roughly constant MSE (≈5 × 10⁻⁶):

| trajectories | 16 | 64 | 256 | 1024 |
|---|---|---|---|---|
| median \|δ\| | 1.6 × 10⁻³ | 1.3 × 10⁻⁴ | 2.6 × 10⁻⁴ | 3.3 × 10⁻⁴ |

Non-monotonic and flat within seed scatter past the smallest case. Sixty-four
times the data buys nothing.

### Standard stability methods act on the wrong direction

Relative to an unregularized baseline at δ = 6.0 × 10⁻⁴:

| method | weight | δ ratio | transverse \|λ₂\| |
|---|---|---|---|
| none | — | 1.00 | 5.7 × 10⁻³ |
| energy penalty | 10⁻² | 0.91 | 5.7 × 10⁻³ |
| energy penalty | 10⁻¹ | 0.87 | 5.8 × 10⁻³ |
| Lipschitz | 10⁻³ | 1.87 | 1.1 × 10⁻² |
| Lipschitz | 10⁻² | 27.4 | 9.8 × 10⁻³ |
| noise injection (pushforward) | 10⁻³ | 1.08 | 5.5 × 10⁻³ |
| noise injection (pushforward) | 10⁻² | 2.73 | 4.5 × 10⁻³ |

None improves δ beyond noise, and Lipschitz regularization at the stronger
weight makes it 27× worse. The λ₂ column matters: these methods do act — the
Lipschitz penalty nearly doubles the transverse multiplier — they act in the
direction that was already contracting.

### Anchoring the cycle

Make both invariants explicit and put the cycle where they say it is:

    ṙ = (r* − r) p(r),   p > 0        cycle exists at r*, attracting
    θ̇ = ω + (r − r*) k(r)            on-cycle angular rate is exactly ω

r* and ω are parameters, supervised against the amplitude and frequency measured
from data by Poincaré section. The transverse rate p stays a free network.

Five seeds each:

| model | median \|δ\| | spread | train MSE | usable horizon |
|---|---|---|---|---|
| free neural ODE | 3.98 × 10⁻⁴ | 3.4e−4 – 7.1e−4 | 5.1 × 10⁻⁶ | 629 T |
| naive polar split | 3.12 × 10⁻⁴ | 1.6e−4 – 2.5e−3 | 6.2 × 10⁻⁶ | 801 T |
| anchored, unsupervised | 3.05 × 10⁻⁴ | 1.6e−4 – 2.1e−2 | 2.5 × 10⁻⁷ | 821 T |
| anchored + supervised | **1.25 × 10⁻⁵** | 4.2e−6 – 2.6e−5 | 1.6 × 10⁻⁷ | 19935 T |

About 32× smaller δ at lower training loss, so nothing is traded for it. Usable
horizon is the rollout length at which phase error reaches π/2, i.e. 1/(4δ).

Structure on its own is worth almost nothing: the naive split and the
unsupervised anchored model land within 30% of the free baseline. What buys the
32× is supervising the two invariants. Note also the spread — unsupervised
anchoring reaches 2.1 × 10⁻², two orders of magnitude worse than its own median,
because nothing pins the cycle.

Two ablations that did not work, kept because they locate the mechanism. The
naive polar split writes θ̇ = ω + k(r) and supervises ω; this constrains nothing,
because the on-cycle rate is ω + k(r*) and k absorbs whatever ω is pinned to.
Anchoring without supervision is unstable: one seed produced δ = −6.4 × 10⁻², a
transverse multiplier of 0.992, and took ten times as long to train.

### What sets the floor

Supervision weight against observation noise on the measured frequency. The
floor column is the error in the measured period itself:

| supervision weight | obs. noise | measurement floor | median \|δ\| | δ / floor |
|---|---|---|---|---|
| 1 | 0 | 1.8 × 10⁻¹⁰ | 1.26 × 10⁻⁵ | 71650 |
| 10² | 0 | 1.8 × 10⁻¹⁰ | 8.21 × 10⁻⁸ | 466 |
| 10⁴ | 0 | 1.8 × 10⁻¹⁰ | 3.75 × 10⁻¹⁰ | 2.1 |
| 10⁴ | 10⁻³ | 9.68 × 10⁻⁸ | 9.68 × 10⁻⁸ | **1.0** |
| 10⁴ | 10⁻² | 9.67 × 10⁻⁷ | 9.67 × 10⁻⁷ | **1.0** |

Weighted hard enough, δ equals the precision with which the frequency can be
measured and nothing else. That is the mechanism in one column.

## Cylinder wake at Re = 100

The oscillator results have phase handed to the model as a coordinate. The wake
is the test of whether any of it survives when phase has to come out of flow
fields.

D2Q9 lattice Boltzmann at 5% blockage, POD by method of snapshots, then a latent
ODE on the coefficients. Modes 1–2 emerge as the shedding pair (95.67% of energy;
harmonics at 1.37/1.35 and 0.75/0.74), and the reference period from POD
coefficient 1 agrees with an independent wake probe to 0.000%, so the latent
phase is the physical one.

**The free model is already good enough here.** At 4000 epochs it reaches
δ ≈ 2 × 10⁻⁵, a usable horizon around 10⁴ shedding periods, past any engineering
need. Reported plainly, because this case was run to test exactly that:

Five seeds per configuration, 4000 epochs:

| model | width | params | med MSE | med \|δ\| | δ spread | horizon |
|---|---|---|---|---|---|---|
| free | 96 | 20264 | 1.01 × 10⁻⁶ | 8.61 × 10⁻⁵ | 37.2× | 2903 P |
| free | 168 | 59648 | 7.31 × 10⁻⁷ | 2.59 × 10⁻⁵ | 6.9× | 9667 P |
| free | 256 | 135944 | 6.92 × 10⁻⁷ | 4.59 × 10⁻⁵ | 6.3× | 5443 P |

What does reproduce is the decoupling. Nearly 7× the parameters buys 30% off the
MSE and nothing on δ, which stays in the 10⁻⁵ band and is non-monotonic in width.
The seed spread is the other half: δ varies 6–37× across seeds at fixed
architecture and data. Long-horizon fidelity here is a lottery that the loss
does not report.

### Phase-anchored latent dynamics

The polar-split model of the previous section fails here — 30× worse δ than free.
The measured cause: orbit radius varies only 0.6% around the cycle, but phase
advance is **0.50% non-uniform**. Forcing θ̇ = ω pointwise is a false constraint,
and the fit trades period accuracy against field accuracy to meet it.

Carrying phase as its own coordinate fixes that. ω is measured rather than
learned, and a Fourier loop Z(φ) absorbs the non-uniformity:

    state (φ, s):   φ̇ = ω,   ṡ = g(s, φ),   a = Z(φ) + s

The rollout period is 2π/ω by construction — no penalty, no weight to tune, and
nothing the optimiser does to Z or g can change it. At 300 epochs, equal
parameter count:

Five seeds, 4000 epochs, against the free baselines above:

| model | params | med MSE | med \|δ\| | δ spread | horizon |
|---|---|---|---|---|---|
| polar-split anchored | 59242 | diverged | 1.20 | 253× | 0 P |
| free (best config) | 59648 | 7.31 × 10⁻⁷ | 2.59 × 10⁻⁵ | 6.9× | 9667 P |
| phase-anchored | 20568 | **5.52 × 10⁻⁷** | **2.75 × 10⁻⁸** | **1.0×** | **9102010 P** |

940× smaller δ than the best free configuration, the lowest MSE of any model
tried, at a seventh the parameters of free-256. The seed spread is the part that
matters most: **1.0×**, against 6–37× for free. Per-seed δ is 2.744, 2.761,
2.741, 2.758, 2.747 × 10⁻⁸. The period stops being a lottery because it is no
longer an outcome of the fit.

That residual 2.75 × 10⁻⁸ is the bias of the period estimator applied to the
rollout, not a property of the model: the model's period is exactly 2π/ω by
construction, and the same five digits appear regardless of seed.

The polar-split model is kept as a failed ablation. It does not merely
underperform — median δ is 1.20, a 120% period error, with some seeds NaN.

### Solver validation, and a retraction

| D | grid | St | vs 0.1643 | period jitter |
|---|---|---|---|---|
| 16 | 480×320 | 0.16643 | +1.30% | 1.7 × 10⁻⁶ |
| 24 | 720×480 | 0.16851 | +2.56% | 5.9 × 10⁻⁷ |
| 32 | 960×640 | 0.16889 | +2.79% | 5.0 × 10⁻⁷ |
| 40 | 1200×800 | 0.16903 | +2.88% | 5.3 × 10⁻⁷ |

St converges to ≈0.169 rather than Williamson's unconfined 0.1643. It took three
hypotheses to find the cause, and the first two are recorded because they are the
ones a reader would try.

Blockage is **not** it: opening the domain from 5% to 1.7% blockage at fixed D
leaves St at ~0.169 and does not move it monotonically.

| ny/D | blockage | St | vs 0.1643 |
|---|---|---|---|
| 20 | 5.00% | 0.16851 | +2.56% |
| 30 | 3.33% | 0.16904 | +2.89% |
| 40 | 2.50% | 0.16930 | +3.05% |
| 60 | 1.67% | 0.16878 | +2.72% |

Compressibility is **not** it either, and moves the wrong way. Holding the
relaxation time at 0.572 and scaling D inversely with inflow speed:

| u_in | D | Ma | St | vs 0.1643 |
|---|---|---|---|---|
| 0.100 | 24 | 0.173 | 0.16851 | +2.56% |
| 0.0667 | 36 | 0.116 | 0.17000 | +3.47% |
| 0.050 | 48 | 0.087 | 0.17113 | +4.16% |

Lowering the Mach number makes the offset worse; extrapolating St against Ma² to
zero Mach gives 0.17175, or +4.54%.

It is **domain truncation**, at both ends:

| upstream | downstream | St | vs 0.1643 |
|---|---|---|---|
| 8 D | 22 D | 0.16851 | +2.56% |
| 8 D | 42 D | 0.16901 | +2.87% |
| 16 D | 34 D | 0.16699 | +1.64% |
| 20 D | 60 D | 0.16587 | **+0.95%** |

Comparing the two runs of equal total length, moving the inlet from 8 D to 16 D
upstream helps more than spending the same cells downstream. The original 8 D /
22 D box was too small at both ends. At 20 D / 60 D the offset is +0.95%, which
is consistent with staircase bounce-back at D = 24.

None of this affects the δ results, which measure a surrogate against the
solver's own period. Period jitter near 5 × 10⁻⁷ is what those need: the
reference period is good to six digits.

**Drag is retracted.** The momentum-exchange estimate gives Cd = 1.05, 0.43,
0.04, −0.20 over D = 16…40, and negative drag is impossible. True drag is ~0.27
in lattice units against a raw link sum of ~15, so the estimate rides on 2%
cancellation, and any systematic error in the link set grows with the perimeter
and overwhelms it. Flagged in the code and excluded from output. Strouhal comes
from the probe and is unaffected.

## Parametric oscillators: where δ actually bites

Everything above fits one operating point, where the model needs a single number.
A parametric operator — the thing a surrogate is usually built for — must produce
ω(μ) at parameter values it never saw. That is the first place δ has a structural
reason to be large, and it is.

van der Pol, ẋ = y, ẏ = ε(1−x²)y − x, with ε trained at 0.5/0.9/1.3/1.7/2.1/2.5,
held out at 0.7/1.1/1.5/1.9/2.3, and extrapolated to 2.9/3.3. The period runs
6.38 → 9.25 across that range and the dependence is nonlinear, so ω(ε) is a real
function to fit. Reference periods agree with published van der Pol values to the
precision those are quoted at, with Poincaré jitter 2 × 10⁻⁸.

Five seeds, 6000 epochs, median |δ|:

| model | width | params | med MSE | train | interp | extrap |
|---|---|---|---|---|---|---|
| free | 96 | 19202 | 1.87 × 10⁻⁴ | 5.23 × 10⁻³ | 3.51 × 10⁻³ | 1.09 × 10⁻¹ |
| free | 150 | 46202 | 9.91 × 10⁻⁵ | 2.48 × 10⁻³ | 2.31 × 10⁻³ | 6.37 × 10⁻² |
| free | 224 | 102146 | 1.07 × 10⁻⁴ | 1.15 × 10⁻³ | 9.45 × 10⁻⁴ | 1.60 × 10⁻² |
| phase-anchored | 96 | 46055 | **1.86 × 10⁻⁵** | **8.49 × 10⁻⁵** | **8.68 × 10⁻⁵** | **5.75 × 10⁻³** |

The same table as usable horizons, which is the part that matters:

| model | params | train | interp | extrap |
|---|---|---|---|---|
| free | 19202 | 48 P | 71 P | **2 P** |
| free | 46202 | 101 P | 108 P | 4 P |
| free | 102146 | 217 P | 265 P | 16 P |
| phase-anchored | 46055 | 2945 P | 2879 P | 43 P |

A parameter-conditioned surrogate decorrelates in 71 periods at held-out
parameters and 2 periods outside the training range, against ~10⁴ periods for the
same architecture at a single operating point. Phase drift is a parametric
problem.

Two things differ from the single-point case and are worth stating plainly.
Capacity **does** buy accuracy here — δ falls from 5.2 × 10⁻³ to 1.2 × 10⁻³ across
the width sweep — because ω(ε) is a function to represent rather than a scalar to
locate. Five times the parameters still lands 13× short of anchoring at half the
size. And extrapolation degrades for every model, anchoring included: 43 periods
at ε outside the training range, against 2879 inside it. Anchoring interpolates a
measured scalar function well and extrapolates it only somewhat better than the
alternative.

## Parametric wake: one operator across Reynolds number

The van der Pol result is an ODE; the single-Re wake is one operating point.
This joins them. D2Q9 lattice Boltzmann at Re = 80…220, trained at
80/100/120/140/160/180, held out at 90/110/130/150/170, extrapolated to
200/220. Solver Strouhal tracks the Williamson correlation with mean offset
+0.09% and spread 0.90%, monotonic in Re. The POD basis is built from the
training Re only, so held-out Re never enters the representation.

Four seeds, 6000 epochs:

| model | params | med MSE | train | interp | extrap |
|---|---|---|---|---|---|
| free | 37392 | 8.50 × 10⁻⁶ | 7.22 × 10⁻⁴ | 7.12 × 10⁻⁴ | 3.36 × 10⁻² |
| free | 176272 | 5.38 × 10⁻⁶ | 3.63 × 10⁻⁴ | 5.36 × 10⁻⁴ | 1.81 × 10⁻² |
| anchored, poly4 ω | 108080 | 8.31 × 10⁻⁶ | **2.69 × 10⁻⁵** | **2.92 × 10⁻⁴** | 1.74 × 10⁻¹ |
| anchored, Williamson ω | 108080 | 8.33 × 10⁻⁶ | 4.79 × 10⁻⁴ | 6.75 × 10⁻⁴ | 1.72 × 10⁻¹ |

As horizons:

| model | train | interp | extrap |
|---|---|---|---|
| free (best) | 688 P | 466 P | 14 P |
| anchored, poly4 ω | **9301 P** | **856 P** | **1 P** |

**The failure case is confirmed.** A free parametric wake operator holds ~500
periods at held-out Re and 14 at extrapolated Re, against ~10⁴ periods for the
same architecture at a single Reynolds number, at unchanged training MSE. Seed
spread is 2–3×, so this is a property of the setting, not a lottery.

### The guarantee covers the phase, not the observable

Anchoring wins in-distribution — 13× at training Re, 1.8× at interpolated Re —
and loses badly outside it, at 1 P against the free baseline's 14 P.

Comparing predicted δ (computable in closed form from the frozen interpolant)
against what the trained model achieves:

| ω interpolant | split | predicted | measured | ratio |
|---|---|---|---|---|
| poly4 | train | 1.36 × 10⁻⁵ | 2.69 × 10⁻⁵ | 1.97 |
| poly4 | interp | 6.89 × 10⁻⁵ | 2.92 × 10⁻⁴ | 4.24 |
| poly4 | extrap | 5.99 × 10⁻³ | 1.74 × 10⁻¹ | **29.0** |
| Williamson | train | 4.52 × 10⁻⁴ | 4.79 × 10⁻⁴ | 1.06 |
| Williamson | interp | 3.68 × 10⁻⁴ | 6.75 × 10⁻⁴ | 1.84 |
| Williamson | extrap | 1.49 × 10⁻³ | 1.72 × 10⁻¹ | **115.6** |

The model meets its predicted δ at training parameters (ratio 1.06) and misses it
by two orders of magnitude outside the training range. The cause is identifiable
rather than speculative. The two interpolants have predicted extrapolation errors
differing by 4×, so if the failure came from ω their measured errors would differ
too. Per seed at Re = 200 and 220 they instead coincide:

| Re | seed | poly4 | Williamson | ratio |
|---|---|---|---|---|
| 200 | 0 | 2.050 × 10⁻¹ | 2.022 × 10⁻¹ | 1.01 |
| 200 | 1 | 1.168 × 10¹ | 1.105 × 10¹ | 1.06 |
| 200 | 2 | 1.075 × 10⁻¹ | 1.081 × 10⁻¹ | 0.99 |
| 220 | 0 | 1.839 × 10⁻¹ | 1.843 × 10⁻¹ | 1.00 |

At interpolated Re the same comparison gives ratios from 0.02 to 1.46, so ω does
matter there. The extrapolation failure is therefore not the frequency.

The reason is structural. The observable is a = Z(φ, μ) + s, and its period
equals 2π/ω(μ) only when the transverse part s decays and Z is a faithful loop.
Both are learned functions of μ. At an unseen μ they are out of distribution, s
does not settle, and the decoded signal carries a spurious component regardless
of how exact the phase is. **Anchoring the phase does not anchor the observable.**

Anchoring also gives up the determinism it had at a single operating point: seed
spread is 29–171× here, against 2–12× for the free baseline, because Z and g are
now parameter-dependent networks rather than fixed ones.

## Classical baselines: the pathology is not neural-specific

Two standard reduced-order models, measured with the same estimator on the same
POD basis. DMD fits a one-step linear map, so its frequency is an explicit
eigenvalue. Operator inference fits the quadratic form adot = c + La + Q(a,a),
the data-driven version of POD-Galerkin; intrusive Galerkin is avoided because
the snapshots live on a cropped window where the discarded boundary terms do not
vanish.

**The mechanism shows up in DMD, from a plain least-squares fit.** The dominant
discrete-time eigenvalue sits on the unit circle to eight decimals:

| Re | \|λ\| | δ |
|---|---|---|
| 80 | 1.00000000 | +4.06 × 10⁻⁷ |
| 100 | 1.00000000 | −1.98 × 10⁻⁸ |
| 130 | 1.00000000 | −2.72 × 10⁻⁷ |
| 180 | 0.99999995 | −7.30 × 10⁻⁶ |

That is the discrete-time signature of the neutrally stable phase direction,
found by a method with no neural network in it. The Floquet argument is not an
artifact of how the surrogates were trained.

### At one operating point, do not use a neural network

| method | median \|δ\| | horizon | stable |
|---|---|---|---|
| phase-anchored | 2.75 × 10⁻⁸ | 9090909 P | 5/5 |
| **DMD (classical)** | **6.92 × 10⁻⁷** | **361247 P** | 6/6 |
| free neural ODE | 2.59 × 10⁻⁵ | 9653 P | 5/5 |
| operator inference | 7.03 × 10⁻³ | 36 P | 4/6 |

A linear least-squares fit gets the frequency **37× better than a trained neural
ODE**, at a small fraction of the cost. Anyone reporting a neural surrogate on a
single-parameter limit cycle without a DMD baseline is reporting against a weak
baseline.

### Parametrically, the ordering reverses

Operators fitted per training Re and interpolated entrywise, the standard
parametric-ROM construction, evaluated at held-out Re:

| method | median \|δ\| | horizon | stable |
|---|---|---|---|
| **phase-anchored** | **2.92 × 10⁻⁴** | **856 P** | 4/4 |
| free neural ODE | 5.36 × 10⁻⁴ | 466 P | 4/4 |
| operator inference | 6.47 × 10⁻³ | 39 P | 2/5 |
| DMD (classical) | 7.64 × 10⁻² | 3 P | 5/5 |

DMD goes from best to worst, losing five orders of magnitude. Interpolating
operator entries does not preserve eigenvalue structure: small entrywise errors
move eigenvalues far, and the frequency is an eigenvalue. Operator inference is
unstable at three of five held-out Re and at both extrapolation points.

### What this means

The period error is not a neural pathology. It is a property of how a method
represents frequency:

- DMD puts it in an eigenvalue. Excellent at a fitted operating point, fragile
  under operator interpolation.
- A learned vector field lets it emerge from field matching. Mediocre at one
  point, but degrades gracefully across parameters.
- Anchoring imposes it from measurement. Best in-distribution, and it fails
  outside the training range for the separate reason given above.

The practical reading is that the choice of surrogate should be made on which
regime is needed, and that δ should be reported in either case, because no
method's training loss reveals it.

### Two properties, both necessary, neither preserved

The parametric DMD result above interpolates operator entries, which is common
practice but not the best a careful person would do. Doing it properly turns out
to isolate the mechanism exactly.

Operators fitted at a single Re have spectral radius 1 to eight decimals — the
neutral phase direction, recovered by least squares with no network involved.
Interpolating between them destroys it:

| | spectral radius |
|---|---|
| fitted directly at each Re | 1.00000001 – 1.00000845 |
| entry-interpolated | 1.10 – **31.05** |

Neutrally-stable operators are a measure-zero set, and linear interpolation
leaves it immediately. A 100-period rollout is ~3000 steps, so this diverges.

Repairing the interpolated spectrum one property at a time, over 13 Reynolds
numbers:

| construction | train | interp | stable |
|---|---|---|---|
| entry interpolation | diverges | diverges | 0/13 |
| mode-matched eigen-interpolation | 3.95 × 10⁻³ (63 P) | 2.21 × 10⁻³ (113 P) | 4/11 |
| generic stabilisation (clip \|λ\|≤1) | 4.81 × 10⁻¹ (1 P) | 4.15 × 10⁻¹ (1 P) | 6/11 |
| + neutrality on the dominant pair | 2.83 × 10⁻² (9 P) | 7.64 × 10⁻² (3 P) | 11/11 |
| **+ imposed measured frequency** | **4.53 × 10⁻⁴ (552 P)** | **3.68 × 10⁻⁴ (679 P)** | 11/11 |
| *frequency interpolant's own error* | *4.53 × 10⁻⁴ (552 P)* | *3.69 × 10⁻⁴ (678 P)* | — |

Read the last two rows together. Imposing neutrality and frequency reproduces the
frequency interpolant's error to four significant figures, 4.525 × 10⁻⁴ against a
predicted 4.526 × 10⁻⁴. The same closed-form predictability the anchored neural
model has, now for a classical linear ROM.

Neither property suffices alone. Neutrality without the frequency stabilises
every rollout but leaves a 3-period horizon. The frequency without neutrality
diverges, because correcting the imaginary part of an eigenvalue does nothing
about a real part sitting at 1.1 to 31. Generic stabilisation, which clips
eigenvalues without knowing which one carries the phase, is worse than useless.

Careful eigen-interpolation is not a way out: matching modes across Re and
interpolating eigenvalues and gauge-fixed eigenvectors still diverges in 7 of 11
cases. Preserving neutrality is genuinely hard, not a matter of being tidier.

**This is the paper's thesis, stated at the right level of generality.** Long-horizon
fidelity of a surrogate of a limit cycle is controlled by two spectral properties
of its phase direction: neutrality, and frequency. Standard construction
preserves neither — not training on field error, not interpolating operators —
and the reported error metric shows neither. Impose both and δ equals the
precision of the frequency estimate, whether the model is a linear ROM or a
neural latent ODE.

Extrapolation fails for every construction here (δ ≈ 1.3, horizon 0 P), matching
the neural result and for the same reason: the mode structure itself is out of
distribution.

## Limitations

Stuart–Landau is a two-dimensional toy with an analytically known cycle. None of
this yet shows the effect dominates in a PDE, that δ is large enough to matter at
engineering horizons, or that anchoring survives a latent phase learned from flow
fields rather than handed over directly. The 200-period horizon is too short:
several configurations never decorrelated within it, so those comparisons are
censored and reported that way. The p = 0.20 correlation is weak evidence of no
relationship, not proof of independence.

The single-point wake shows the diagnostic transferring while the warning does
not: δ ≈ 2 × 10⁻⁵ is not what limits that surrogate. A Re = 100 wake is
low-dimensional and near-harmonic, and 8 POD modes on one noise-free trajectory
is the easiest version of the task.

The parametric results supply the missing failure case, but on an ODE rather than
a PDE. What has *not* been shown is a parametric **flow** surrogate with large δ —
one operator across a range of Reynolds number, where the Strouhal curve is known
independently. That is the experiment that would join the two halves, and it is
the next one to run.

Extrapolation is a genuine limitation rather than a detail. Every model degrades
outside the training range, anchoring included (43 periods against 2879 inside).
Anchoring interpolates a measured scalar function well; it does not confer
extrapolation.

The van der Pol splits are one-dimensional in parameter. Real operators condition
on several parameters at once, where the frequency surface is harder to fit and
the held-out points are further from their neighbours in a way a 1-D sweep does
not capture.

## Install and run

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
```

Pilot, a few minutes on a laptop:

```bash
python src/t0_falsify.py --mode pilot --epochs 3000 --periods 200
```

Full sweeps on Slurm (RIT SPORC):

```bash
sbatch --array=0-74 cluster/t0_array.sh all
sbatch --array=0-19 cluster/t0_fix_array.sh
sbatch --array=0-17 cluster/t0_anchor_array.sh
python src/analyze.py "results/all/*.json"
```

Cylinder wake:

```bash
sbatch --array=0-3 cluster/cyl_res_array.sh
sbatch --array=0-3 cluster/cyl_blockage.sh
sbatch cluster/cyl_snapshots.sh
sbatch cluster/cyl_latent_job.sh
```

## Layout

| path | contents |
|---|---|
| `src/t0_falsify.py` | ground truth, neural ODE, Floquet and drift diagnostics, scaling and regularizer sweeps |
| `src/t0_fix.py` | anchored parametrization and the naive-split ablation |
| `src/t0_anchor_sweep.py` | supervision weight against observation noise |
| `src/cylinder_lbm.py` | D2Q9 lattice Boltzmann cylinder wake at Re = 100 |
| `src/cyl_latent.py` | POD, free / polar-split / phase-anchored latent ODEs |
| `src/analyze.py`, `src/harvest.py` | aggregation and the prediction tests |
| `cluster/` | Slurm array scripts |
