# Two spectral properties govern long-horizon fidelity in surrogates of oscillatory flows, and standard practice preserves neither

*Draft targeting Nature Machine Intelligence. Supersedes `workshop_paper_draft.md`.*

## Abstract

Autoregressive surrogates of oscillatory flows lose long-horizon accuracy
through accumulated phase drift, not amplitude error [1]. We give the mechanism,
and show that the methods used to stabilise long rollouts act on a direction that
was never at fault.

By time-translation invariance a stable limit cycle has one Floquet multiplier at
unity along the orbit, the rest inside the unit circle: amplitude error
self-corrects, phase error has no restoring force. Long-horizon fidelity is set
by one scalar, the relative period error δ, measurable from a ten-period rollout
and predicting observed drift to a median ratio of 1.0000.

Nothing in standard practice sees δ. The three stability methods in common use
leave it unchanged or worsen it — the worst by 27× — while contracting a
direction already stable. Nor is the effect neural: least-squares dynamic mode
decomposition recovers the same neutral multiplier, and the Fourier neural
operator that fits best is the one that cannot finish a rollout.

Two spectral properties govern the outcome, neither sufficient alone. Impose both
and δ equals the precision of the frequency estimate, so a usable horizon is
computable before a model is fitted — one falling from 10⁴ periods to 14 when a
wake operator is conditioned on Reynolds number.

## 1. Introduction

Rollout error curves in the neural-PDE literature look alike. Error grows, then
flattens near a plateau. The plateau has usually been read as saturation of a
generic accumulation process, and the remedies proposed for it are generic too:
train on longer rollouts, inject noise, penalise the Jacobian. Recent work on
bluff-body wakes shows the error is not generic at all — almost all of it is
accumulated phase drift, with the attractor geometry reproduced to a fraction of
a percent [1]. That reframes the problem without resolving it, because a
description of the error is not yet an account of what produces it, what sets
its size, or which interventions can reach it.

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

## 1.1 Relation to existing work

**The phenomenon has been reported.** Samanta and Behera [1] show that for
bluff-body wakes at Re = 100–800 the autoregressive rollout error of a
convolutional-autoencoder LSTM is 95–98% pure phase error, peaking at the
shedding frequency; the network reproduces the attractor geometry to within
0.15% in amplitude and traverses it at slightly the wrong rate, while one-step
validation error looks essentially perfect. We take that observation as
established and do not re-claim it. What their result leaves open is why it
happens, whether it is specific to learned models, what sets the size of the
error, and what to do about it. Those are the questions this paper answers.

**Floquet theory has been applied to the same problem, with the opposite
prescription.** Li and Matzakos [2] bound long-horizon neural-ODE error by
certifying that the learned return map contracts, using a differentiable loss on
the monodromy determinant, and note that phase drift between the two clocks
persists. The bound is correct and the diagnosis is compatible with ours, but
the remedy is aimed at the wrong direction. A stable limit cycle has one
multiplier exactly at unity along the orbit; contraction penalties act on the
transverse multipliers, which were already inside the unit circle. Section 2
measures what that costs, and the measurement is not neutral: the stronger
Lipschitz weight leaves the transverse multiplier nearly doubled and the period
error 27 times worse.

**The rollout-stability literature is large and does not measure period error.**
Multi-step and pushforward training [3], noise injection, spectral-norm and
Jacobian regularisation [4, 5], and spectral refinement [6] all target rollout
divergence, and all are evaluated on rollout error, spectra, or divergence
rates. We are not aware of a study that reports what any of them does to the
frequency of a periodic solution. Section 2 does, for three of them, and finds
that the best of them leaves δ unchanged while the worst degrades it by more
than an order of magnitude.

**Neutral eigenvalues are foundational in Koopman analysis** [7], where unit
modulus is the signature of sustained oscillation. The contribution here is not
that the neutral direction exists but that its accuracy is the binding
constraint on a surrogate's usable horizon, is measurable in ten periods, and is
absent from every loss and every basis-selection criterion in standard use.

**Parametric reduced-order models are known to lose stability under
interpolation,** and stability-preserving interpolation is an established line of
work [8, 9]. Section 5 sharpens the target: for a limit cycle, generic
stabilisation is not enough, because clipping eigenvalues without knowing which
one carries the phase leaves the dominant pair untouched. Neutrality on that
pair and a measured frequency are both required, and imposing them reproduces
the frequency interpolant's own error to within 0.03%.

**Neural operators** [10, 11] are evaluated here because the claim is aimed at
them. Section 3 finds the effect intact on a Fourier neural operator and, in the
configuration that fits best, most severe.

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
mode decomposition [12], no network involved — recovers the neutral multiplier:

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

**On a Navier–Stokes field, not a latent space.** Everything above acts on a
cylinder wake compressed to eight POD coefficients, which leaves open whether
the period error is a property of the problem or of low-dimensional latent
dynamics. We close that by measuring δ on the two-dimensional Navier–Stokes
equations directly, in the forcing geometry of the FNO benchmark itself [10],
with the operator acting on the full 128 × 128 vorticity field.

Locating a usable case is part of the result. The benchmark ships at ν = 10⁻³,
10⁻⁴ and 10⁻⁵, and a split-half refit disagrees with itself by 10⁻³ to 10⁻¹ at
all three, so a δ quoted on them would measure nothing. Sweeping viscosity
across and beyond that range leaves the flow steady for ν ≥ 1.9 × 10⁻³ and
aperiodic for ν ≤ 1.8 × 10⁻³, with a narrow window at ν = 1.7 × 10⁻³ where the
attractor is a clean limit cycle: split-half 4.8 × 10⁻⁹ over 51 periods,
enstrophy steady to 8 × 10⁻⁴. **All three benchmark settings sit past that
transition.** Rollouts below are 4002 steps — 100 periods — projected onto an
8-mode basis and fitted with the same estimator as everywhere else.

| variant | stable | med train MSE | med \|δ\| | seed spread | horizon |
|---|---|---|---|---|---|
| residual | 3/3 | **2.14 × 10⁻⁶** | **1.20 × 10⁻²** | 12× | **21 P** |
| direct | 3/3 | 3.24 × 10⁻⁵ | 2.40 × 10⁻⁴ | **2×** | **1042 P** |
| + noise injection | 3/3 | 9.69 × 10⁻⁶ | 1.89 × 10⁻³ | 148× | 132 P |

The residual operator fits **fifteen times better and has a fifty times worse
period**: a usable horizon of 21 cycles against 1042. On the wake the same
comparison was 2.4× in loss and 52× in δ, so moving to the field widens the loss
advantage while the period disadvantage holds. Nothing differs between those two
rows except whether the network predicts the field or the increment.

The direct parameterisation is again the most reproducible, at 2× seed spread
against 10× for the same variant on the wake. Two systems of very different
character agree on it, which makes it a design recommendation rather than an
idiosyncrasy of one problem.

Noise injection we report as undetermined here. Its median δ is 6.4× better than
the residual baseline, but its three seeds span 148× — 2.1 × 10⁻⁵, 1.9 × 10⁻³
and 3.1 × 10⁻³ — while on the wake the same intervention moved the median by
1.09× with a spread of 890×. Across both systems its effect on δ is not
resolvable at these seed counts. That is consistent with a method acting
elsewhere than on the phase, and it is not evidence for it.

One property of this case belongs in the open rather than left for a reader to
find: eight POD modes capture 100.000% of the energy, so the attractor is
effectively eight-dimensional even though the state has 16384 components and the
operator acts on all of them. That is a feature of the object, not a weakness of
the test. A limit cycle is a low-dimensional attractor by construction, which is
exactly why the Floquet argument applies to it; a system whose attractor were
genuinely high-dimensional would lie outside the scope this paper claims.

## 4. The objective, not the model class

`linear-sgd` is exactly DMD's hypothesis class — one matrix — fitted by Adam on
a 25-step rollout instead of by least squares on one step. It is 6× better. We
predicted the opposite, expecting gradient descent to lose to the closed-form
solution and implicate the optimiser.

Optimized DMD [13] tests the alternative explanation with a third algorithm. It fits
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
tracks the Williamson correlation [14] with mean offset +0.09% and spread 0.90%.

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

## References

*All entries verified against indexed records, September 2026. Two remain worth
a final check at proof stage: [16], whose page range is from a secondary source,
and the arXiv preprints [1], [2] and [5], which may acquire journal versions
before this paper appears.*

[1] **[v]** S. Samanta and S. Behera, "Autoregressive rollout error in
latent-space reduced-order models of bluff-body wakes is accumulated phase
drift," arXiv:2608.07189 (2026). doi:10.48550/arXiv.2608.07189

[2] **[v]** Z. Li and N. M. Matzakos, "Long-time trajectory approximation via
SA-NODEs: model predictive and Floquet strategies," arXiv:2608.10738 (2026).
doi:10.48550/arXiv.2608.10738

[3] **[v]** J. Brandstetter, D. Worrall and M. Welling, "Message passing neural
PDE solvers," *Int. Conf. Learning Representations* (2022). arXiv:2202.03376.
doi:10.48550/arXiv.2202.03376

[4] **[v]** M. McCabe, P. Harrington, S. Subramanian and J. Brown, "Towards
stability of autoregressive neural operators," *Transactions on Machine Learning
Research* (2023). arXiv:2306.10619; OpenReview RFfUUtKYOG.

[5] **[v]** F. Nie and Y. Suzuki, "Jacobian-adaptive weighting for stability:
enhancing long-term rollout of neural partial differential equation solvers via
spatially-adaptive regularization," arXiv:2603.05538 (2026).
doi:10.48550/arXiv.2603.05538

[6] **[v]** P. Lippe, B. S. Veeling, P. Perdikaris, R. E. Turner and
J. Brandstetter, "PDE-Refiner: achieving accurate long rollouts with neural PDE
solvers," *Advances in Neural Information Processing Systems* **36** (2023).
arXiv:2308.05732

[7] **[v]** S. L. Brunton, M. Budišić, E. Kaiser and J. N. Kutz, "Modern Koopman
theory for dynamical systems," *SIAM Review* **64**, 229–340 (2022).
doi:10.1137/21M1401243

[8] **[v]** D. Amsallem and C. Farhat, "Interpolation method for adapting
reduced-order models and application to aeroelasticity," *AIAA Journal* **46**,
1803–1813 (2008). doi:10.2514/1.35374

[9] **[v]** M. Geuss, H. Panzer and B. Lohmann, "STABLE — a stability algorithm
for parametric model reduction by matrix interpolation," *Mathematical and
Computer Modelling of Dynamical Systems* **22**, 307–322 (2016).
doi:10.1080/13873954.2016.1198383

[10] **[v]** Z. Li, N. B. Kovachki, K. Azizzadenesheli, B. Liu, K. Bhattacharya,
A. M. Stuart and A. Anandkumar, "Fourier neural operator for parametric partial
differential equations," *Int. Conf. Learning Representations* (2021).
arXiv:2010.08895. doi:10.48550/arXiv.2010.08895

[11] **[v]** L. Lu, P. Jin, G. Pang, Z. Zhang and G. E. Karniadakis, "Learning
nonlinear operators via DeepONet based on the universal approximation theorem of
operators," *Nature Machine Intelligence* **3**, 218–229 (2021).
doi:10.1038/s42256-021-00302-5

[12] **[v]** P. J. Schmid, "Dynamic mode decomposition of numerical and
experimental data," *Journal of Fluid Mechanics* **656**, 5–28 (2010).
doi:10.1017/S0022112010001217

[13] **[v]** T. Askham and J. N. Kutz, "Variable projection methods for an
optimized dynamic mode decomposition," *SIAM Journal on Applied Dynamical
Systems* **17**, 380–416 (2018). doi:10.1137/M1124176

[14] **[v]** C. H. K. Williamson, "Oblique and parallel modes of vortex shedding
in the wake of a circular cylinder at low Reynolds numbers," *Journal of Fluid
Mechanics* **206**, 579–627 (1989). doi:10.1017/S0022112089002429
