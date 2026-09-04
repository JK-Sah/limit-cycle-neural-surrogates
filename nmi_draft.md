# Two spectral properties govern long-horizon fidelity in surrogates of oscillatory flows, and standard practice preserves neither

*Draft targeting Nature Machine Intelligence. Supersedes `workshop_paper_draft.md`.*

## Abstract

Reduced-order and learned surrogates of oscillatory systems lose accuracy over
long rollouts. The usual response is a stability mechanism — an energy penalty,
a Lipschitz constraint, noise injection during training — and we show these act
on the wrong direction. A stable limit cycle has one Floquet multiplier exactly
equal to unity along the orbit, forced by time-translation invariance, with the
rest strictly inside the unit circle. Amplitude error therefore decays on its
own; phase error has no restoring force and accumulates without bound. Long-horizon
behaviour is set by one scalar, the relative period error δ, which is
measurable from a short rollout and predicts decorrelation far beyond it.

Across 75 trained models on a Stuart–Landau oscillator, observed phase drift
matches the rate predicted from δ with median ratio 1.0000. δ cannot be read off
the training loss: at fixed architecture, data and objective it spreads by up to
6 × 10⁴× across random seeds alone, it is unchanged by 64× more data,
and left alone or made 27× worse by the three standard stability methods. The
result is not a property of neural networks: on a cylinder wake, dynamic mode
decomposition recovers the neutral multiplier to |λ| = 1.00000000 by least
squares alone, and seven surrogate families — continuous, discrete,
recurrent, linear, and classical — span six orders of magnitude in δ while
their training losses do not distinguish them. A recurrent model reaches a
lower training loss than two that work and has no usable limit cycle at all.

Two properties control the outcome, and both must be imposed. Enforcing
neutrality of the phase direction on a parametric reduced-order model makes
every rollout bounded but leaves a 3-period horizon; adding a measured
frequency reproduces the frequency interpolant's own error to within 0.03%,
4.503 × 10⁻⁴ against 4.504 × 10⁻⁴ predicted. Neither alone suffices.
The same pairing explains the classical results: optimized DMD with harmonic
locking, which fixes both by construction, reaches δ ≈ 1.3 × 10⁻⁸ and is
insensitive to basis and truncation, while exact DMD varies by 16,000× with
basis choice alone.

The practical consequence appears in parametric operators, which is what
surrogates are built for. Conditioning a wake operator on Reynolds number drops
its usable horizon from ~10⁴ shedding periods to 466 at held-out Re and 14
outside the training range, with training loss unchanged.

The cost is asymmetric, and predictable. Over rollouts of 30,000 shedding
periods, time-averaged statistics from a surrogate with an attracting cycle stay
flat to within 10% across a 3000-fold increase in rollout length, while
phase-resolved quantities follow 2|sin(πδN)| and first decorrelate at
N = 1/(4δ) — matched per seed to within one sampling interval. Surrogates
without an attractor forfeit both: exact DMD's |λ| = 1.00001312 compounds over
9 × 10⁵ steps and its covariance error grows by a factor of 10¹¹. We report
where our own remedy fails: anchoring the phase does not anchor the decoded
observable, and outside the training range the anchored model is worse than
doing nothing.

We also locate where the argument itself stops. On a quasi-periodic attractor
the single-period fit becomes ill-posed — it disagrees with itself by 38% on one
record, against 10⁻⁹ once the oscillators lock — so a δ quoted there measures
nothing. Splitting a record and refitting separates them cleanly at every record
length we tested, and is worth doing before any δ is trusted. The natural
two-frequency generalisation of the error law does not hold, and we do not claim
it.

## 1. Introduction

Rollout error curves in the neural-PDE literature look alike. Error grows, then
flattens near a plateau. The plateau is usually read as saturation of a generic
accumulation process, and the remedies proposed for it are generic too.

For a system on a limit cycle the shape has a specific cause, and it is
available in closed form. Consider the monodromy matrix of a periodic orbit of
an autonomous ODE. One of its eigenvalues is exactly 1, corresponding to
displacement along the orbit. This follows from time-translation invariance and
holds for any autonomous system with a periodic solution, learned or not. For a
stable cycle the remaining multipliers lie inside the unit circle.

Two consequences follow and they point in opposite directions. A perturbation
transverse to the cycle decays at a rate set by the subdominant multiplier, so
amplitude error is self-correcting. A perturbation along the cycle neither grows
nor decays, so phase error is never corrected. If a surrogate's period differs
from the truth by a relative amount δ, phase error grows as

    Δθ(t) ≈ −2π δ t / T

without bound, until the prediction is antiphase with the truth and relative L²
error saturates near √2. The plateau is decorrelation, not saturation.

This makes a prediction sharp enough to be wrong: long-horizon error should be
fully accounted for by a single scalar measurable from a short rollout, and
mechanisms acting on the transverse directions should not help. Both hold, and
the second holds more strongly than we expected.

## 2. The mechanism

We begin with Stuart–Landau in Cartesian form, where the cycle radius √μ, the
angular frequency ω − βμ and the transverse multiplier e^(−2μT) are known in
closed form. Models are neural ODEs with a tanh MLP vector field trained by RK4
rollout on short segments; numbers are medians over five seeds and evaluation
rollouts are 200 periods.

**The phase law is exact.** Over 75 trained models the ratio of observed phase
drift to the rate predicted from the measured period error has median 1.0000
and range 1.000–1.033. Median residual from a straight-line fit to Δθ(t) is
1.4 × 10⁻³ rad. As a check on the diagnostic itself, the computed leading
multiplier deviates from 1 by at most 1.2 × 10⁻¹⁰.

**Phase is unbounded, amplitude is not.** At 200 periods, median |Δθ| is 0.82 rad
and the worst case 52.0 rad, against a median maximum amplitude error of
3.3 × 10⁻³ and a cycle radius of 0.707.

**The training loss does not see it.** Across 35 runs spanning a 64× range of
dataset size, log(train MSE) and log|δ| have ρ = −0.22 (p = 0.20). Two runs with
identical architecture and data, and indistinguishable loss, differ by two
orders of magnitude in δ:

| trajectories | width | seed | train MSE | δ | decorrelation |
|---|---|---|---|---|---|
| 64 | 32 | 3 | 9.1 × 10⁻⁶ | +1.4 × 10⁻³ | 179 T |
| 64 | 32 | 4 | 7.2 × 10⁻⁶ | +1.1 × 10⁻⁵ | > 200 T |

**Scaling does not fix it.** Median |δ| by training set size, at roughly constant
MSE: 1.6 × 10⁻³ (16 trajectories), 1.3 × 10⁻⁴ (64), 2.6 × 10⁻⁴ (256),
3.3 × 10⁻⁴ (1024). Non-monotonic and flat within seed scatter past the smallest
case.

**Standard stability methods act on the transverse direction.** Relative to an
unregularized baseline at δ = 6.0 × 10⁻⁴:

| method | weight | δ ratio | transverse \|λ₂\| |
|---|---|---|---|
| none | — | 1.00 | 5.7 × 10⁻³ |
| energy penalty | 10⁻² | 0.91 | 5.7 × 10⁻³ |
| energy penalty | 10⁻¹ | 0.87 | 5.8 × 10⁻³ |
| Lipschitz | 10⁻³ | 1.87 | 1.1 × 10⁻² |
| Lipschitz | 10⁻² | 27.4 | 9.8 × 10⁻³ |
| noise injection | 10⁻³ | 1.08 | 5.5 × 10⁻³ |
| noise injection | 10⁻² | 2.73 | 4.5 × 10⁻³ |

None improves δ beyond seed noise, and the Lipschitz penalty at the stronger
weight makes it 27× worse. The λ₂ column is the point. These methods do act —
the Lipschitz penalty nearly doubles the transverse multiplier — and what they
act on is the direction that was already contracting.

## 3. It is not a neural pathology

Flow past a cylinder at Re = 100, D2Q9 lattice Boltzmann, POD by method of
snapshots, latent dynamics on the coefficients. Modes 1–2 emerge as the shedding
pair carrying 95.7% of the energy, and the period measured from POD coefficient 1
agrees with an independent wake probe to 0.000%, confirming the latent phase is
the physical one.

Fitting a one-step linear map to those coefficients by least squares — dynamic
mode decomposition, no network involved — recovers the neutral multiplier:

| Re | \|λ\| | δ |
|---|---|---|
| 80 | 1.00000000 | +2.52 × 10⁻⁸ |
| 100 | 1.00000000 | −2.82 × 10⁻⁸ |
| 130 | 1.00000000 | −2.14 × 10⁻⁷ |
| 180 | 0.99999995 | −6.57 × 10⁻⁶ |

The Floquet structure is a property of the problem, recovered by a method with
no training in it.

Seven surrogate families on the same 8-mode basis, same data, same estimator,
five seeds, with harmonic-locked optimized DMD (§4) carried at the top for
reference:

| model | params | med MSE | med \|δ\| | seed spread | horizon |
|---|---|---|---|---|---|
| optDMD, harmonic-locked | — | — | 1.32 × 10⁻⁸ | — | 18937960 P |
| ode-deep | 114440 | 3.77 × 10⁻⁶ | 1.13 × 10⁻⁵ | 2.2 × 10⁴× | 22225 P |
| ode-mlp | 20264 | 1.94 × 10⁻⁶ | 4.19 × 10⁻⁵ | 38× | 5972 P |
| linear-sgd | 64 | 2.81 × 10⁻⁴ | 7.56 × 10⁻⁵ | 21× | 3306 P |
| ar-direct | 20264 | 3.32 × 10⁻⁴ | 9.67 × 10⁻⁵ | 71× | 2584 P |
| ar-residual | 20264 | 6.43 × 10⁻⁵ | 1.42 × 10⁻⁴ | 3.5× | 1760 P |
| exact DMD | 64 | — | 4.58 × 10⁻⁴ | 1.0× | 546 P |
| gru | 31304 | 4.88 × 10⁻⁵ | 4.49 × 10¹ | 6.0 × 10⁴× | **0 P** |

Six orders of magnitude in δ, and the loss column does not order them in any way
a practitioner could use. The relationship is worth stating precisely, because it
is not simply absent. Taking each architecture as one observation — the
defensible unit, since seeds within an architecture are not independent —
median training loss and median |δ| across the twelve configurations here
correlate at ρ = +0.62 (p = 0.033): families that fit better do tend to have
smaller δ. That trend is useless as a tool, because the spread *within* a single
architecture at fixed data and objective runs from 3.5× to 6 × 10⁴×, swamping
it. Loss is weakly informative about a model class and carries no information
about a model. The GRU reaches a
lower training loss than `linear-sgd` and `ar-direct`, both of which work, and
has no usable limit cycle — its period is wrong by a factor of forty. A
recurrent hidden state carries dynamics of its own and drifts over a long
rollout, and nothing in the objective sees it.

Exact DMD's seed spread is 1.0× because it is a deterministic closed-form fit.
Every learned model spreads between 3.5× and 6.0 × 10⁴×, so a single-seed δ says
almost nothing about a learned surrogate.

**The architectures the field actually deploys.** Every model above acts on POD
coefficients. Neural operators do not, and they are what operator learning is
built on, so the claim has to be tested on one rather than beside it. We train a
Fourier neural operator on the wake *fields*, u(x, y, t) → u(x, y, t + Δt) on the
full 2 × 192 × 96 grid, and measure δ by rolling it out for 3215 steps —
100 periods — projecting onto the same 8-mode POD basis and fitting coefficient 1
with the same estimator. Only the model changes.

| training objective | variant | stable | med train MSE | med \|δ\| | seed spread | horizon |
|---|---|---|---|---|---|---|
| 8-step rollout | residual | 0/1 | **2.88 × 10⁻⁶** | diverged at step 645 | — | **0 P** |
| 25-step rollout | residual | 4/5 | 7.3 × 10⁻⁶ | 4.48 × 10⁻⁴ | 141× | 558 P |
| 25-step rollout | direct | **5/5** | 1.58 × 10⁻⁵ | **8.61 × 10⁻⁶** | **10×** | **29033 P** |
| 25-step + noise injection | residual | 5/5 | 1.92 × 10⁻⁵ | 4.90 × 10⁻⁴ | 890× | 510 P |

The shortest objective produces the **lowest training loss of the four** and is
the only configuration that cannot complete a rollout at all, diverging
two-thirds of the way through; lengthening the objective to 25 steps costs a
factor of two in training loss and buys a rollout that survives. That is §4's
argument — the objective horizon rather than the model class — reappearing on an
architecture chosen by a different community, and in a sharper form, because
here the better-fitting model is the one that breaks.

Rollout survival is itself seed-dependent and equally invisible to the loss. One
of the five residual seeds diverges at step 2010 having trained to 6.56 × 10⁻⁶,
against 5.99 × 10⁻⁶ for the best seed that survives to the end. Nothing in the
training curve separates them.

The direct parameterisation is the strongest learned model in this study on both
counts that matter: median δ = 8.6 × 10⁻⁶, better than any latent model here, and
a seed spread of 10×, against 141× for the residual form and up to 6 × 10⁴×
elsewhere in the table. It reaches that while fitting twice as poorly.

Noise injection is the informative case, and it is informative precisely because
its effect is confined. It rescues the diverging seed — 5 of 5 rollouts complete
rather than 4 of 5 — and leaves the median period error where it was, 4.90 × 10⁻⁴
against 4.48 × 10⁻⁴, a ratio of 1.09. A stability method restored stability and
did not touch the phase. That is the transverse/tangential split of §2 appearing
on a neural operator, and the reason the whole set of such methods leaves δ
unchanged in that table too. It does have one cost: the seed spread widens from
141× to 890×, so what the method buys in worst-case survival it gives back in
reproducibility.

## 4. The objective, not the model class

`linear-sgd` is exactly DMD's hypothesis class — one matrix — fitted by Adam on
a 25-step rollout instead of by least squares on one step. It is 6× better. We
predicted the opposite, expecting gradient descent to lose to the closed-form
solution and implicate the optimiser.

Optimized DMD tests the alternative explanation with a third algorithm. It fits
the same exponential model against the entire trajectory by Levenberg–Marquardt
on the exponents, with linear coefficients projected out. On a single-Re basis
at 8 modes its free form beats exact DMD by **26,700×**, and the harmonic-locked
form introduced below by 34,800×. The objective horizon, not the
model class and not the optimiser, is what recovers the frequency: one-step
least squares does not penalise accumulated phase error and a multi-step fit
does.

Optimized DMD also exposes the second property. In its free form, p independent
(σ, ω) pairs, nothing constrains σ to zero. At 16 modes its analytic frequency
is excellent (δ = 1.3 × 10⁻⁶) while its 100-period rollout is destroyed
(δ = 3.1 × 10⁻¹, horizon 1 period), because unconstrained growth modes dominate
the reconstruction. Locking the modes to harmonics of one fundamental with
σ = 0 removes the failure entirely: δ = 1.3–1.9 × 10⁻⁸ across every basis and
truncation tested.

## 5. Neutrality and frequency, both necessary

The clearest statement of the mechanism comes from parametric reduced-order
models. Operators fitted at a single Reynolds number have spectral radius 1 to
eight decimals. Interpolating between them entrywise — standard parametric-ROM
practice — gives spectral radius 1.10 to 31.05. Neutrally stable operators form
a measure-zero set and linear interpolation leaves it immediately.

Repairing the interpolated spectrum one property at a time, across 13 Reynolds
numbers:

| construction | train | interp | stable |
|---|---|---|---|
| entry interpolation | diverges | diverges | 0/13 |
| mode-matched eigen-interpolation | 3.95 × 10⁻³ | 2.53 × 10⁻³ | 4/11 |
| generic stabilisation (clip non-dominant \|λ\| ≤ 1) | 4.07 × 10⁻¹ | 3.91 × 10⁻¹ | 6/11 |
| + neutrality on the dominant pair | 2.83 × 10⁻² | 7.64 × 10⁻² | 11/11 |
| **+ imposed measured frequency** | **4.50 × 10⁻⁴** | **3.65 × 10⁻⁴** | 11/11 |
| *frequency interpolant's own error* | *4.50 × 10⁻⁴* | *3.66 × 10⁻⁴* | — |

The last two rows agree to within 0.03%. Imposing both properties
makes δ equal to the precision of the frequency estimate and nothing else — and
because the rollout period is then 2π/ω by construction, δ is computable in
closed form before any model is fitted. A practitioner can select the frequency
interpolant by cross-validation on measured frequencies alone and know the
usable horizon in advance.

The middle row is a real intervention rather than a straw man, and worth
stating precisely: generic stabilisation does clip, and it clips every
non-dominant eigenvalue with a positive growth rate back onto the unit circle.
What it cannot do is identify which eigenvalue carries the phase, so it leaves
the dominant pair untouched and the spectral radius at 2.96. Neutrality is the
step that brings the radius to exactly 1.

Neither property suffices. Neutrality without the frequency stabilises every
rollout and leaves a 3-period horizon. Frequency without neutrality diverges,
since correcting an eigenvalue's imaginary part does nothing about a real part
at 1.1 to 31. Careful eigen-interpolation, the fair version a reviewer would
propose, still diverges in 7 of 11 cases.

## 6. Where it bites: parametric operators

At a single operating point the effect is real and small. A well-trained free
latent ODE reaches δ ≈ 2.6 × 10⁻⁵ on the Re = 100 wake, a horizon near 10⁴
shedding periods, past any engineering need. We report this because the case was
run to test exactly that.

Surrogates are built to cover a parameter range. Conditioning on Reynolds
number, with the POD basis built from training Re only so held-out values never
enter the representation:

| model | params | train | interp | extrap |
|---|---|---|---|---|
| free | 37392 | 7.22 × 10⁻⁴ | 7.12 × 10⁻⁴ | 3.36 × 10⁻² |
| free | 176272 | 3.63 × 10⁻⁴ | 5.36 × 10⁻⁴ | 1.81 × 10⁻² |
| anchored (poly ω) | 108080 | 2.69 × 10⁻⁵ | 2.92 × 10⁻⁴ | 1.74 × 10⁻¹ |

In horizons, the free operator holds 466 periods at interpolated Re and 14
outside the training range, against ~10⁴ at a single Reynolds number, with
training MSE unchanged and seed spread 2–3×. The same pattern appears on a
parametric van der Pol oscillator: 71 periods at held-out ε and 2 outside the
range. Conditioning on a parameter costs roughly a factor of 20 in horizon, and
the loss does not report it.

## 7. What it costs

A diagnostic earns its place by predicting something a practitioner computes.
The claim is a split rather than a slope. Phase-invariant quantities — the
time-mean field, RMS fluctuation, the covariance that stands in for Reynolds
stresses — should not depend on where in the cycle a surrogate believes it is.
Phase-resolved quantities — forecasting a given instant, data assimilation,
phase-locked actuation — should decorrelate at N ≈ 1/(4δ).

We roll seven surrogates out to 30,000 shedding periods. Ground truth at
arbitrary horizon comes from phase-wrapping the reference trajectory, exact to
6 × 10⁻¹⁴ here because the measured period jitter is 5 × 10⁻⁷.

**Phase-resolved error follows 2|sin(πδN)|.** A phase slip of Δθ = 2πδN produces
relative RMS error 2|sin(Δθ/2)|. This does not saturate. It first reaches √2 at
N = 1/(4δ) and returns to zero whenever the slip passes a multiple of 2π.
Observed against predicted:

| model | median | p10 | p90 |
|---|---|---|---|
| ode-mlp | 1.036 | 0.969 | 1.103 |
| ar-residual | 1.023 | 0.966 | 1.112 |
| ar-direct | 1.034 | 0.988 | 1.699 |
| ode-deep | 1.080 | 0.705 | 1.135 |

The recurrence is directly visible: `ode-mlp` reaches 1.960 at 10,000 periods and
falls back to 0.529 at 20,000, rather than staying saturated. We quote that as
qualitative evidence only. Near a recurrence the prediction is a difference of
large phases and becomes acutely sensitive to δ, which spans 1.5–4.8 × 10⁻⁵
across seeds here; a median-δ prediction at that point is not meaningful, and the
quantitative agreement in the table above excludes predictions below 0.05 for the
same reason.

Per seed, the first crossing of √2 matches the predicted 1/(4δ) with median ratio
1.52 across 14 runs, on a horizon grid spaced about 3× apart — agreement to
within one sampling interval.

**Phase-invariant statistics stay flat, provided the model has an attractor.**
Covariance error at 10 periods against 30,000, a 3000-fold increase in rollout
length:

| model | N = 10 | N = 30000 | growth | |
|---|---|---|---|---|
| ode-mlp | 6.2 × 10⁻³ | 6.7 × 10⁻³ | **1.1×** | nonlinear |
| ode-deep | 6.1 × 10⁻³ | 6.5 × 10⁻³ | **1.1×** | nonlinear |
| ar-residual | 7.3 × 10⁻³ | 7.8 × 10⁻³ | **1.1×** | nonlinear |
| ar-direct | 3.9 × 10⁻² | 4.0 × 10⁻² | **1.0×** | nonlinear |
| linear-sgd | 4.2 × 10⁻² | 9.9 × 10⁻¹ | 23.7× | linear |
| exact DMD | 3.5 × 10⁻² | 3.8 × 10⁹ | **1.1 × 10¹¹×** | linear |
| gru | 1.9 × 10⁻¹ | 1.1 × 10¹⁰ | 5.8 × 10¹⁰× | no cycle |

The exceptions are the argument, not counterexamples to it. A linear surrogate
has no attractor, so nothing bounds its amplitude except |λ| itself, and exact
DMD's |λ| = 1.00001312 compounds over the 9.7 × 10⁵ steps in this rollout to an
amplitude factor of 3 × 10⁵. That accounts for the direction and rough scale of
the covariance growth, though not its precise value — 3.8 × 10⁹ measured against
~10¹¹ from amplitude alone, the shortfall reflecting that the growth is not
uniform across modes. Neutrality is what keeps amplitude bounded; without it,
even the phase-invariant statistics fail. The same two properties that govern the
period govern what survives a long rollout.

The usable statement: time-averaged statistics from a surrogate with an
attracting limit cycle are trustworthy indefinitely; anything phase-resolved has
a shelf life of 1/(4δ) cycles, computable from a ten-period rollout; and a linear
surrogate has neither guarantee unless |λ| = 1 is imposed.

## 8. Where our remedy fails

Anchoring the phase to a measured frequency wins in-distribution — 13× at
training parameters, 1.8× at held-out ones — and loses badly outside the
training range, at 1 period against the free baseline's 14.

The cause is identifiable rather than assumed. We ran two frequency
interpolants whose predicted extrapolation errors differ by 4×. If ω were
responsible their measured errors would differ too. Per seed at Re = 200 and
220 they instead coincide to within 1–6% (2.050 × 10⁻¹ against 2.022 × 10⁻¹;
1.839 × 10⁻¹ against 1.843 × 10⁻¹), while at interpolated Re they differ by up
to 50×.

The explanation is structural. The observable is a = Z(φ, μ) + s, and its period
equals 2π/ω(μ) only when the transverse part s decays and Z is a faithful loop.
Both are learned functions of μ, both are out of distribution at unseen μ, and
the decoded signal then carries a spurious component however exact the phase is.
**Anchoring the phase does not anchor the observable.** The anchored model also
gives up the reproducibility it has at one operating point: seed spread is
29–171× here against 2–12× for the free baseline.

## 9. Where the argument stops

The whole analysis assumes a limit cycle: one neutral direction, one period, one
scalar. A quasi-periodic attractor has two incommensurate frequencies and no
single period, so the diagnostic cannot apply as stated. Rather than assert the
boundary we locate it, using two Stuart–Landau oscillators with diffusive
coupling swept across the locking transition. The regime is read off the rate at
which the two oscillators' phases separate:

| eps | 0.0 | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 | 0.8 |
|---|---|---|---|---|---|---|---|
| phase drift | 0.77 | 0.75 | 0.66 | 0.49 | **0.00** | **0.00** | **0.00** |

**The boundary is one of well-posedness.** The single-period fit always latches
onto the stronger frequency as its fundamental; what changes across the
transition is whether the result means anything. Fitting the two halves of one
record separately:

| eps | regime | half-vs-half instability | δ from the fit | δ measured properly |
|---|---|---|---|---|
| 0.00 | quasi-periodic | **3.8 × 10⁻¹** | 6.08 × 10⁻¹ | 1.9 × 10⁻³, 1.8 × 10⁻³ |
| 0.10 | quasi-periodic | 1.3 × 10⁻⁴ | 6.11 × 10⁻¹ | 5.4 × 10⁻³, 4.6 × 10⁻³ |
| 0.30 | quasi-periodic | 1.5 × 10⁻⁵ | 1.16 × 10⁻⁴ | 1.0 × 10⁻⁴, 1.2 × 10⁻⁴ |
| 0.50 | locked | **1.8 × 10⁻⁹** | 2.84 × 10⁻⁴ | the same number |
| 0.80 | locked | **9.6 × 10⁻⁹** | 8.80 × 10⁻⁴ | the same number |

Six to eight orders of magnitude separate the two regimes here. At eps = 0 the fit disagrees
with itself by 38% on a single record, and its reported δ of 0.61 is not a
measurement of anything — both true frequencies are in fact learned to
2 × 10⁻³. Once locked the fit is stable to 10⁻⁹ and its δ agrees exactly with a
direct measurement, so §§2–8 apply unchanged.

This yields a check worth running before any δ is quoted: **split the record,
refit, compare.** What makes it usable is that the separation is a threshold
rather than a ratio. Sweeping the observation window from 200 to 1600 time units,
the gap between the two regimes ranges from 3.7 to 7.6 orders of magnitude — so
the size of the separation is a property of the record, not of the attractor, and
should not be quoted as though it were the latter. What does not move is where
the two populations sit. A locked record refits to within 10⁻⁸ at every window
tested (worst case 9.6 × 10⁻⁹); a quasi-periodic one never does better than
10⁻⁶ (best case 1.1 × 10⁻⁶). The populations are separated by a factor of 118
and never overlap, so a threshold near 10⁻⁷ classifies every case correctly
regardless of how much data is in hand — which is the situation a practitioner
is actually in.

**The obvious generalisation does not work.** Replacing one scalar with two,
err = √(Σₖ wₖ · 4 sin²(π δₖ Nₖ)), gives observed/predicted 1.04 at eps = 0 but
3.19, 0.81 and 0.68 at eps = 0.1, 0.2 and 0.3, and is worse than the
single-frequency law in the locked regime (5.34 against 1.39 at eps = 0.8). One
agreeing case is not a law, and we do not claim the extension. A weight-imbalance
account of which quasi-periodic cases survive is also refuted: the two frequency
families carry 0.44–0.56 of the energy at every coupling tested. Chaotic
attractors are out of scope.

## 10. Discussion

Long-horizon fidelity of a surrogate of a limit cycle is controlled by two
spectral properties of its phase direction: neutrality, and frequency. Standard
construction preserves neither. Training on field error does not, because the
loss is an L² reconstruction criterion and δ is orthogonal to it. Interpolating
operators does not, because neutrality is measure-zero. Fitting one step ahead
does not, because a one-step objective cannot see accumulated phase.

Methods that impose both properties succeed regardless of whether they are
learned or classical: harmonic-locked optimized DMD, a repaired parametric ROM,
and a phase-anchored latent model all reach δ within an order of magnitude of
their frequency estimate's own precision. Methods that impose neither fail in
characteristic ways — exact DMD by being 16,000× basis-sensitive, the GRU by
having no cycle at all, free optDMD by growing an unconstrained mode.

The two properties also decide what survives a long rollout, not just the
period. Neutrality bounds the amplitude, so a surrogate that has it keeps its
time-averaged statistics indefinitely and loses only phase; a surrogate that
lacks it loses both. This is why the linear models in §7 fail at statistics that
the nonlinear ones hold flat, despite the linear models having smaller δ.

The practical recommendations are short. Check that the period fit is
well-posed — split the record, refit, compare — since on a quasi-periodic
attractor a δ can be reported that means nothing. Then report δ; it is measurable
from ten periods and predicts decorrelation at 1/(4δ). Do not read long-horizon fidelity
off a validation error. Report more than one seed for a learned surrogate, since
δ spreads by up to 6 × 10⁴× at fixed architecture and data. Match the surrogate
to the question: for time-averaged statistics almost anything with an attractor
will do, and for phase-resolved work the horizon is fixed in advance by δ. If
the frequency can be measured, impose it, and impose neutrality with it —
expecting the benefit in-distribution only.

## 11. Limitations

Every system here has a low-dimensional attractor, and every one that §§2–8 draw
conclusions from is periodic. Chaotic attractors have no periodic orbit and the
argument does not transfer to them at all. For the quasi-periodic side §9 locates
the boundary and gives a test for being on the wrong side of it, but supplies no
replacement law once there — the natural two-frequency generalisation was tried
and does not hold.

The wake is 2D at Reynolds numbers where the physical flow becomes
three-dimensional above roughly 190, so the extrapolation points are numerically
valid and physically artificial. Reference-period jitter grows from 5 × 10⁻⁷ at
Re = 80 to 3 × 10⁻⁴ at Re = 220, so δ below that floor is not resolvable at the
extrapolation points. Parameter sweeps are one-dimensional; real operators
condition on several parameters at once, where the frequency surface is harder
to fit and held-out points sit further from their neighbours.

The relationship between training loss and δ is a correlation-versus-causation
trap and we have fallen into one version of it already. Within a fixed
architecture and objective, loss and δ are unrelated: on the Stuart–Landau
scaling runs ρ = −0.22 at p = 0.15. Pool across heterogeneous conditions and a
positive association appears — over all 75 oscillator runs including the
regularised ones, ρ = +0.30 at p = 0.009; over the wake architectures taken one
per family, ρ = +0.62 at p = 0.033. Neither is evidence that loss predicts δ. It
is the ordinary fact that a model class which fits badly also tracks the period
badly, and pooling classes manufactures the trend. Quoting the pooled number as
though it measured a within-model relationship would be wrong in one direction,
and quoting the within-model number as though it showed global independence
would be wrong in the other. What the claim actually rests on is the spread at
fixed everything: 3.5× to 6 × 10⁴× in δ across seeds of one architecture on one
dataset with one objective. No loss-based criterion can select against that.

Two measurement issues are worth recording. Our first period estimator, phase
regression on the analytic signal, is unbiased for a pure sinusoid and biased at
1.2 × 10⁻⁵ for harmonic-rich signals. It was replaced by a Fourier fit with
unknown fundamental, validated against signals of known period at 10⁻¹⁰ and
seeded from the periodogram so it cannot be handed the answer. Everything in
§§2–4 and §§7–9 is measured with the replacement, and re-measurement changed no
ordering and no conclusion. Separately, our momentum-exchange drag estimate is
retracted: it gives Cd = 1.05, 0.43, 0.04, −0.20 over D = 16…40, and negative
drag is impossible. Strouhal number comes from a wake probe, is unaffected, and
tracks the Williamson correlation with mean offset +0.09% and spread 0.90%.

Every table has been re-measured with the replacement, and the pattern across
them is itself informative. The per-Reynolds DMD table in §3 was the one case
where the old estimator's bias exceeded the quantity being measured — its δ
values run from 10⁻⁸ to 10⁻⁵, taken against a reference period the estimator
supplies — and there the δ entries moved by 10% to a factor of nine while |λ|
was unchanged in every case to eight decimals. The eigenvalue is read off the
fitted operator and never touches the estimator, so the claim that table
supports was never at risk; the individual δ values at the 10⁻⁸ level simply
were not resolved by the old measurement. Where the measured quantity sits well
above the bias, re-measurement changed almost nothing: the parametric-ROM table
in §5 moved by under 0.5% in its two decisive rows, and the parametric-wake
numbers in §6 by under 10% in thirteen of fifteen entries, with no ordering
disturbed anywhere.

---

## Figures to produce

1. **The phase law.** Δθ(t) over 200 periods for ~10 models, overlaid with the
   line predicted from each model's measured δ. Inset: histogram of
   observed/predicted slope centred on 1.000.
2. **Decomposition.** Relative L² against rollout length for one model, split
   into phase and amplitude. Amplitude flat, phase linear, total saturating
   at √2.
3. **Loss is blind.** Scatter of log(train MSE) against log|δ| across all seven
   architectures, coloured by family, with the two correlation coefficients
   annotated.
4. **Two properties.** Eigenvalue plane showing the interpolated spectrum
   (radius up to 31), after generic stabilisation, after neutrality, and after
   neutrality plus frequency, with the horizon under each.
5. **Parametric collapse.** Usable horizon against parameter for free and
   anchored models on the wake, training Re marked, showing degradation between
   training points and collapse outside the range.

## Still open

- Flexible FSI, where the structure sets its own timescale rather than
  inheriting one — not run, and the strongest remaining system class.
- A theory for the quasi-periodic side of the boundary. §9 shows where the
  single-scalar description fails and gives a test for it, but the obvious
  two-frequency replacement was tried and does not hold.
- The DeepONet direct parameterisation trains 33× worse than its own residual
  form, and the cause is structural rather than a training failure: tripling the
  epochs and tripling the learning rate both leave the loss at 1.18 × 10⁻²
  unchanged, while giving the model a learnable mean field halves it to
  5.7 × 10⁻³ and improves δ by 52×. The residual form remains 16× better still.
  Worth noting that the two operator families split in opposite directions here —
  the direct parameterisation is the FNO's best configuration and the DeepONet's
  worst — which is a real architectural difference and not yet explained.
- All three tables that predated the estimator correction have been re-run
  (`rom_wake_v2`, `rom_parametric_v2`, `parwake_v2`). No conclusion or ordering
  changed anywhere; only §3's δ column moved materially, and its |λ| column not
  at all.
- Repo is private until acceptance. Before submitting, check NMI's current
  code-availability policy and decide the reviewer-access route (private repo
  shared with the editor, anonymised snapshot, or a Zenodo reviewer link) —
  Nature Portfolio expects code reachable during review, not just on
  publication. Mint a Zenodo DOI from a release at acceptance.
- The public README still reports the pre-correction architecture-breadth and
  truncation tables, has no optDMD section, and carries a Limitations section
  written before the parametric-wake work existed. It needs a sync pass before
  anyone is pointed at the repo.
