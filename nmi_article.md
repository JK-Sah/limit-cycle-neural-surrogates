# Two spectral properties govern long-horizon fidelity in surrogates of oscillatory flows, and standard practice preserves neither

J. K. Sah

Department of Mechanical and Industrial Engineering, Rochester Institute of
Technology, Rochester, NY, USA. e-mail: js8472@rit.edu

---

## Abstract

A stable limit cycle has one Floquet multiplier exactly at unity along the
orbit, forced by time-translation invariance, the rest inside the unit circle.
Amplitude error in a learned surrogate therefore self-corrects and phase error
never does, so long-horizon fidelity reduces to one scalar: the relative period
error δ, measurable from a ten-period rollout. Across 75 models δ predicts
observed drift to a median ratio of 1.0000. Nothing in standard practice sees
it: of three stability methods in common use none improves δ, and a Lipschitz
penalty degrades it 27-fold while contracting an already-stable direction. Nor
is the effect neural, since least-squares dynamic mode decomposition recovers
the same neutral multiplier. What does reach δ is the training objective. A
rollout segment spanning a full period improves it 81-fold on a Navier–Stokes
operator for under 2× in fit, and imposing neutrality with a measured frequency
drives it to the precision of that frequency estimate.

---

Rollout error curves in the neural-PDE literature look alike. Error grows, then
flattens near a plateau. That plateau has usually been read as saturation of a
generic accumulation process, and the remedies proposed for it are generic too:
train on longer rollouts, inject noise, penalise the Jacobian. Recent work on
bluff-body wakes shows the error is not generic at all. For a convolutional-
autoencoder LSTM at Re = 100–800, Samanta and Behera<sup>1</sup> find the
autoregressive rollout error is 95–98% pure phase error, peaking at the shedding
frequency, while the network reproduces the attractor geometry to within 0.15% in
amplitude and its one-step validation error looks essentially perfect. We take
that observation as established and do not re-claim it. What it leaves open is
why the error takes that form, whether it is specific to learned models, what sets its size, and which interventions reach it.

For a system on a limit cycle the answer is available in closed form. Consider
the monodromy matrix of a periodic orbit of an autonomous ODE. One of its
eigenvalues is exactly 1, corresponding to displacement along the orbit. This
follows from time-translation invariance and holds for any autonomous system with
a periodic solution, learned or not. For a stable cycle the remaining multipliers
lie strictly inside the unit circle.

Two consequences follow, and they point in opposite directions. A perturbation
transverse to the cycle decays at a rate set by the subdominant multiplier, so
amplitude error is self-correcting. A perturbation along the cycle neither grows
nor decays, so phase error is never corrected. If a surrogate's period differs
from the truth by a relative amount δ, phase error grows as Δθ(t) ≈ −2πδt/T
without bound, until the prediction is antiphase with the truth and relative L²
error saturates near √2. The plateau is decorrelation, not saturation.

This is sharp enough to be wrong in a useful way: it predicts that long-horizon
error is fully accounted for by a single scalar measurable from a short rollout,
and that mechanisms acting on the transverse directions cannot help. Both hold,
and the second more strongly than we expected.

Floquet theory has been brought to this problem before, with the opposite
prescription. Li and Matzakos<sup>2</sup> bound long-horizon neural-ODE error by
certifying that the learned return map contracts, using a differentiable loss on
the monodromy determinant, and note that phase drift between the two clocks
persists. The bound is correct and the diagnosis is compatible with ours, but the
remedy is aimed at the wrong direction: contraction penalties act on the
transverse multipliers, which were already inside the unit circle. We measure
what that costs. The broader rollout-stability literature — multi-step and
pushforward training<sup>3</sup>, noise injection, spectral-norm and Jacobian
regularisation<sup>4,5</sup>, spectral refinement<sup>6</sup> — is evaluated on
rollout error, spectra or divergence rates, and we are not aware of a study that
reports what any of it does to the frequency of a periodic solution.

Neutral eigenvalues are foundational in Koopman analysis<sup>7</sup>, where unit
modulus is the signature of sustained oscillation. The contribution here is not
that the neutral direction exists but that its accuracy binds a surrogate's
usable horizon, is measurable in ten periods, and is absent from every loss and
basis-selection criterion in standard use. Parametric reduced-order models are
likewise known to lose stability under interpolation<sup>8,9</sup>; we sharpen
the target by showing generic stabilisation is insufficient for a limit cycle. Neural operators<sup>10,11</sup> are tested
here because the claim is aimed at them.

## Results

### The phase law is exact

We begin with the Stuart–Landau oscillator, where the cycle radius, angular
frequency and transverse multiplier are known in closed form (Methods). Numbers
are medians over five seeds and evaluation rollouts run 200 periods.

Over 75 trained models the ratio of observed phase drift to the rate predicted
from the measured period error has median 1.0000 and range 1.000–1.033
(Fig. 1). Median residual from a straight-line fit to Δθ(t) is 1.4 × 10⁻³ rad. As
a check on the diagnostic itself, the computed leading multiplier deviates from 1
by at most 1.2 × 10⁻¹⁰.

Phase is unbounded and amplitude is not. At 200 periods, median |Δθ| is 0.82 rad
and the worst case 52.0 rad, against a median maximum amplitude error of
3.3 × 10⁻³ on a cycle of radius 0.707.

The training loss does not see any of this. Across the 40 runs of the scaling
sweep, spanning a 64× range of dataset size, log(train MSE) and log|δ| have
ρ = −0.21 (p = 0.19). Two runs with identical architecture, data and indistinguishable loss differ by
two orders of magnitude in δ: at 64
trajectories and width 32, seed 3 trains to 9.1 × 10⁻⁶ and gives
δ = +1.4 × 10⁻³, a 179-period horizon, while seed 4 trains to 7.2 × 10⁻⁶ and
gives δ = +1.1 × 10⁻⁵, still correlated past 200 periods. Scaling does not fix it: median |δ| by training-set size, at roughly constant
MSE, runs
1.6 × 10⁻³ (16 trajectories), 1.3 × 10⁻⁴ (64), 2.6 × 10⁻⁴ (256), 3.3 × 10⁻⁴
(1024) — non-monotonic and flat within seed scatter past the smallest case.

### Stability methods act on the transverse direction

Relative to an unregularised baseline at δ = 6.0 × 10⁻⁴, none of three stability
methods in common use improves δ beyond seed noise, and the Lipschitz penalty at
the stronger weight makes it 27.4× worse (Extended Data Table 1). The transverse multiplier column is the point: these
methods do act — the Lipschitz penalty
nearly doubles |λ₂|, from 5.7 × 10⁻³ to 1.1 × 10⁻² — and what they act on is the
direction that was already contracting.

### It is not a neural pathology

Flow past a cylinder at Re = 100, with latent dynamics on POD coefficients
(Methods). The period from POD coefficient 1 agrees with an independent wake probe to
0.000%, confirming the latent phase is the physical one.

Fitting a one-step linear map to those coefficients by least squares — dynamic
mode decomposition<sup>12</sup>, no network involved — recovers the neutral multiplier to within 5 × 10⁻⁸ of unity at every Reynolds
number tested (Extended Data Table 2). The Floquet structure belongs to the problem, not to the training.

Across seven surrogate families on the same 8-mode basis, data, estimator and
five seeds, δ spans six orders of magnitude and the loss column does not order them in any usable way (Extended Data Table 3).
A GRU reaches lower training loss than both `linear-sgd` and `ar-direct`, which
work, and has no usable limit cycle at all: its period is wrong by a factor of
forty. A recurrent hidden state carries dynamics of its own and drifts over a
long rollout, and nothing in the objective sees it.

The relationship between loss and δ is worth stating precisely, because it is not
simply absent (Fig. 3). Taking each configuration as one observation — the
defensible unit, since seeds within a configuration are not independent — median
training loss and median |δ| correlate at ρ = +0.62 (p = 0.033) across the twelve
wake configurations, and at ρ = +0.45 (p = 0.095) across all fifteen including
the Navier–Stokes operators below. Families that fit better do tend to have smaller δ, but the trend is useless as
a tool: the spread *within* a single configuration at fixed data and objective
runs from 1.9× to 6.0 × 10⁴×, swamping it. Loss is weakly informative about a
model class and carries no information about a model. Exact DMD's seed spread is
1.0× only because it is a deterministic closed-form fit; every learned model
spreads by at least 1.9×, so a single-seed δ says little about a learned
surrogate.

### Neural operators, and the objective that closes the orbit

Every model above acts on POD coefficients. Neural operators do not, and they
are what operator learning is built on, so the claim has to be tested on one. We train a Fourier neural operator on the wake *fields*,
u(x, y, t) → u(x, y, t + Δt) on the full 2 × 192 × 96 grid, and measure δ by
rolling it out for 3215 steps (100 periods), projecting onto the same 8-mode POD
basis and fitting coefficient 1 with the same estimator. Only the model changes.

The shortest objective produces the lowest training loss of the four
configurations and is the only one that cannot complete a rollout, diverging
two-thirds of the way through (Extended Data Table 4); lengthening it from 8 to
25 steps costs 2.5× in training loss and buys a rollout that survives. The
objective horizon, not the model class, governs the outcome, and in a sharper
form than in the latent models, because the better-fitting model is the one that
breaks. Two further algorithms separate the objective from the model class and the
optimiser. `linear-sgd` is DMD's hypothesis class, one matrix, fitted by Adam on a 25-step
rollout rather than by least squares on one step, and is 6× better; optimized
DMD<sup>13</sup>, fitted against the whole trajectory, beats exact DMD by
26,700× on a single-Re basis.

**The objective must close the orbit.** Lengthening the segment until it spans a
full shedding period is the largest single improvement in δ we obtain. At 32.15
snapshots per period a 32-step objective covers 0.995 of a revolution; over
seven seeds it gives δ = 1.93 × 10⁻⁴ against the 25-step objective's 4.48 ×
10⁻⁴, and every rollout completes rather than four of five. On the Navier–Stokes
field, where a period is 40.02 snapshots, going from 25 to 40 steps improves δ
by 81×, from 1.20 × 10⁻² to 1.47 × 10⁻⁴, moving the usable horizon from 21 periods to 1696 (Extended Data Tables 4, 5).
Training loss pays 1.3× and 1.8×. The magnitudes track how far each fell short
of a period: 0.62 on Navier–Stokes, 0.78 on the wake. A segment that closes the
orbit constrains the phase accumulated over one revolution; one that stops short
leaves it free.

Noise injection is the informative case, because its effect is confined. It
rescues the diverging seed — 5 of 5 rollouts complete rather than 4 of 5 — and
leaves the median period error where it was, 4.90 × 10⁻⁴ against 4.48 × 10⁻⁴, a
ratio of 1.09: a stability method restored stability and did not touch the
phase. Its one cost is reproducibility, the seed spread widening from 141× to
890×.

Everything above acts on a wake compressed to eight POD coefficients, leaving
open whether the period error belongs to the problem or to low-dimensional
latent dynamics. We close that by measuring δ on the two-dimensional
Navier–Stokes equations directly, in the forcing geometry of the FNO
benchmark<sup>10</sup>, on the full 128 × 128 vorticity field.

Locating a usable case is part of the result. The benchmark ships at ν = 10⁻³,
10⁻⁴ and 10⁻⁵, and a split-half refit disagrees with itself by 10⁻³ to 10⁻¹ at
all three, so a δ quoted on them would measure nothing (Fig. 6). Sweeping
viscosity leaves the flow steady for ν ≥ 1.9 × 10⁻³ and aperiodic for ν ≤ 1.8 ×
10⁻³, with a narrow window at ν = 1.7 × 10⁻³ where the attractor is a clean
limit cycle: split-half 4.8 × 10⁻⁹ over 51 periods.

Measured there at 25 steps, the residual operator fits fifteen times better than
the direct one and has a fifty times worse period: a horizon of 21 cycles
against 1042 (Extended Data Table 5). On the wake the same comparison was 2.2×
in loss and 52× in δ. Nothing differs but whether the network predicts the field or the increment. The
direct parameterisation is also the most reproducible on both systems, 2× seed
spread against 10× on the wake. Noise injection we report as undetermined: across both systems its
effect on δ is not resolvable at these seed counts.

One property of this case belongs in the open: eight POD modes capture 100.000%
of the energy, so the attractor is effectively eight-dimensional though the
state has 16,384 components. That is a feature of the object rather than a
weakness of the test — a limit cycle is low-dimensional by construction, which
is why the Floquet argument applies.

### Neutrality and frequency are both necessary

The clearest statement of the mechanism comes from parametric reduced-order
models. Operators fitted at a single Reynolds number have spectral radius 1 to
eight decimals. Interpolating between them entrywise, which is standard
parametric-ROM practice, gives spectral radius 1.03 to 31.05: neutrally stable operators form a measure-
zero set, and linear interpolation leaves it
immediately (Fig. 4).

Repairing the interpolated spectrum one property at a time across 13 Reynolds
numbers (Extended Data Table 6) shows that imposing both makes δ equal to the
precision of the frequency estimate and nothing else. The two final rows agree to
within 0.03%: 4.503 × 10⁻⁴ measured against 4.504 × 10⁻⁴ predicted by the
frequency interpolant's own error. Because the rollout period is then 2π/ω by
construction, δ is computable in closed form before any model is fitted. A
practitioner can select the frequency interpolant by cross-validation on measured
frequencies alone and know the usable horizon in advance.

Generic stabilisation is a real intervention rather than a straw man: it clips
every non-dominant eigenvalue with a positive growth rate back onto the unit
circle, but cannot identify which one carries the phase, so it leaves the
dominant pair untouched and the radius at 2.96. Neither property suffices alone.
Neutrality without the frequency stabilises every rollout and leaves a 3-period
horizon; frequency without neutrality diverges, since correcting an imaginary
part does nothing about a real part at 1.1 to 31. Careful
eigen-interpolation, the fair version a reviewer would propose, still diverges in
7 of 11 cases.

### Conditioning on a parameter costs the horizon

At a single operating point the effect is real and small: a well-trained free
latent ODE reaches δ ≈ 2.6 × 10⁻⁵ on the Re = 100 wake, a horizon near 10⁴ periods, past any engineering need.

Surrogates are built to cover a parameter range. Conditioning on Reynolds number,
with the POD basis built from training Re only so held-out values never enter the
representation, the free operator holds 433 periods at interpolated Re and 9 outside the
training range, against ~10⁴ at a single Reynolds number, while median training loss varies by only 1.5× across the models compared and
seed spread stays at 1.5–16× (Fig. 5, Extended Data Table 7). The same pattern appears on a parametric van der Pol oscillator: 71 periods at
held-out ε and 2 outside the range for the smallest of three model sizes, 265
and 16 for the largest. Conditioning on a parameter costs roughly a
factor of 20 in horizon, and the loss does not report it.

Our own anchoring remedy fails here, and the cause is identifiable rather than
assumed. Anchoring the phase to a measured frequency wins in-distribution, 15× at
training parameters and 1.8× at held-out ones, and loses badly outside the
training range, at 1 period against the free baseline's 9. Two frequency interpolants whose predicted extrapolation errors differ by 4×
produce measured errors coinciding to within 1–3% at Re = 200 and 220, while
differing by up to 5× at interpolated Re, so ω is not responsible.
The cause is structural: the observable is a = Z(φ, μ) + s, whose period equals
2π/ω(μ) only when the transverse part s decays and Z is a faithful loop. Both
are learned functions of μ and both are out of distribution at unseen μ, so the
decoded signal carries a spurious component however exact the phase is.
Anchoring the phase does not anchor the observable.

### What δ predicts, and where the diagnostic stops

A diagnostic earns its place by predicting something a practitioner computes, and
the claim here is a split (Fig. 2). Phase-invariant quantities — the time-mean
field, RMS fluctuation, the covariance standing in for Reynolds stresses — should
not depend on where in the cycle a surrogate believes it is, while phase-resolved
quantities should decorrelate at N ≈ 1/(4δ). Rolling seven
surrogates out to 30,000 shedding periods, phase-resolved error follows
2|sin(πδN)| with per-model median observed/predicted ratios of 1.02–1.08
(Extended Data Table 8), and the first crossing of √2 matches the predicted 1/(4δ) to a median ratio of
1.52 across 14 runs. Covariance error over a 3000-fold increase in rollout
length grows by 1.0–1.1× for every model with an attractor (Extended Data
Table 9).

The exceptions are the argument rather than counterexamples. A linear surrogate
has no attractor, so nothing bounds its amplitude but |λ| itself,
and exact DMD's |λ| = 1.00001312 compounds over 9.7 × 10⁵ steps to a covariance
error of 3.8 × 10⁹. Neutrality keeps amplitude bounded; without it even the
phase-invariant statistics fail.

The whole analysis assumes a limit cycle: one neutral direction, one period, one
scalar. A quasi-periodic attractor has two incommensurate frequencies and no
single period, so the diagnostic cannot apply as stated. Rather than assert the
boundary we locate it, using two Stuart–Landau oscillators with diffusive
coupling swept across the locking transition (Extended Data Table 10). The boundary is one of well-posedness: the single-period fit always latches onto
the stronger frequency as its fundamental, and what changes across the
transition is whether the result means anything. At ε = 0 the fit disagrees with itself by 38%
on a single record and its reported δ of 0.61 measures nothing, while both true
frequencies are in fact learned to 2 × 10⁻³. Once locked, the fit is stable to
10⁻⁹ and its δ agrees exactly with a direct measurement.

This yields a check worth running before any δ is quoted: split the record,
refit, compare. The separation is a threshold rather than a ratio. Sweeping the observation window from 200 to 1600 time units, the margin between the worst locked record and the best quasi-periodic one
ranges from 400× to 3,400×, so the size of the gap is a property of the record, not of
the attractor. What does not move is where the two populations sit: a locked
record refits to within 10⁻⁸ at every window tested and a quasi-periodic one
never better than 1.1 × 10⁻⁶, separated by a factor of 118 without overlap.

The obvious generalisation does not work. Replacing one scalar with two,
err = √(Σₖ wₖ · 4 sin²(π δₖ Nₖ)), gives observed/predicted 1.04 at ε = 0 but
3.19, 0.81 and 0.68 at ε = 0.1, 0.2 and 0.3, and is worse than the single-frequency law once locked. A weight-imbalance
account of which cases survive is also refuted: the two frequency families carry
0.44–0.56 of the energy at every coupling.

## Discussion

Long-horizon fidelity of a surrogate of a limit cycle is controlled by two
spectral properties of its phase direction: neutrality, and frequency. Standard
construction preserves neither. Training on field error does not, because the
loss is an L² reconstruction criterion and δ is orthogonal to it. Interpolating
operators does not, because neutrality is measure-zero. Fitting one step ahead
does not, because a one-step objective cannot see accumulated phase.

Methods that impose both properties succeed whether learned or classical. Harmonic-locked optimized DMD<sup>13</sup>, a repaired
parametric ROM, and a phase-anchored latent model all reach δ within an order of
magnitude of their frequency estimate's own precision. Methods that impose
neither fail in characteristic ways: exact DMD by being 17,000× basis-sensitive,
the GRU by having no cycle at all, free optimized DMD by growing an unconstrained mode that leaves its analytic
frequency excellent and its 100-period rollout destroyed.

The two properties also decide what survives a long rollout, not only the period.
Neutrality bounds the amplitude, so a surrogate that has it keeps its
time-averaged statistics indefinitely and loses only phase, while a surrogate
that lacks it loses both. This is why the linear models fail at statistics the nonlinear ones hold flat,
despite having smaller δ.

The practical recommendations are short. Check that the period fit is well-posed
by splitting the record and refitting, since on a quasi-periodic attractor a δ
can be reported that means nothing. Then report δ; it is measurable from ten
periods and predicts decorrelation at 1/(4δ). Train on a rollout segment long enough to close the orbit: one that spans a full
period is worth up to 81× in δ and costs under 2× in fit. Do not read long-
horizon fidelity off a validation error. Report more than one seed. Match the surrogate to the question: for time-averaged statistics almost
anything with an attractor will do, and for phase-resolved work the horizon is
fixed in advance by δ. If the frequency can be measured, impose it together with
neutrality, expecting the benefit in-distribution only.

The scope is narrower than the mechanism might suggest: every system here has a
low-dimensional attractor, and every one the central results rest on is
periodic. Chaotic attractors have no periodic orbit and the argument does not transfer.
For the quasi-periodic side we locate the boundary and give a test for it, but
supply no replacement law once there.
The wake is two-dimensional at Reynolds numbers where the physical flow becomes
three-dimensional above roughly 190, so the extrapolation points are numerically valid and physically artificial;
reference-period jitter grows to 3 × 10⁻⁴ at Re = 220, below which δ is not
resolvable. Parameter sweeps are
one-dimensional; real operators condition on several parameters at once.

The relationship between training loss and δ is a correlation-versus-causation
trap we fell into before correcting it. Within a fixed architecture and
objective the two are unrelated (ρ = −0.22, p = 0.15 over 45 unregularised
oscillator runs); pool across heterogeneous conditions and a positive
association appears (ρ = +0.30, p = 0.009 over all 75). Neither is evidence that
loss predicts δ: a model class which fits badly also tracks the period badly,
and pooling classes manufactures the trend. What the claim rests on is the spread at fixed everything, which no
loss-based criterion can select against.

## Methods

**Oscillator system.** The Stuart–Landau oscillator is integrated in Cartesian
coordinates,

    ẋ = (μ − r²)x − (ω − βr²)y,   ẏ = (ω − βr²)x + (μ − r²)y,   r² = x² + y²,

with μ = 0.5, ω = 1.0 and β = −0.5. The cycle radius is √μ = 0.70711, the
angular frequency on the cycle is Ω = ω − βμ = 1.25, the period is
T = 2π/Ω = 5.02655 and the transverse Floquet multiplier is
e^(−2μT) = 6.55 × 10⁻³, all in closed form. Reference trajectories are produced
with an adaptive Dormand–Prince (DOP853) integrator at rtol = atol = 10⁻¹².
Training data are short segments from randomised initial conditions on and near
the cycle; the scaling sweep spans 16, 64, 256 and 1024 trajectories.

**Wake system.** Flow past a circular cylinder is simulated with a D2Q9
lattice-Boltzmann solver, cylinder diameter D = 24 lattice units, inlet velocity
u_in = 0.1 in lattice units, kinematic viscosity ν = u_in D/Re and relaxation
time τ = 3ν + 0.5, over Re = 80–220. The domain is 80D × 20D with the cylinder
centred 20D from the inlet, giving 60D of downstream wake; this geometry was
chosen after domain truncation, not blockage or compressibility, was identified
as the source of an earlier Strouhal offset. Runs are 140,000 steps with snapshots taken every 45 steps from the second half
of the record, well past the transient. Snapshots are cropped to a wake window
running 2D upstream to 14D downstream and ±4D across the centreline, then taken
at stride 2, giving the 2 × 192 × 96 velocity field the neural operators act on;
the POD basis is built from the same field. Fields are compressed by proper orthogonal
decomposition using the method of snapshots. Modes 1–2 form the shedding pair
and carry 95.7% of the energy at Re = 100. Latent surrogates act on the leading
8 POD coefficients unless stated otherwise, at 32.15 snapshots per shedding
period, so a 100-period rollout is 3215 steps. For parametric studies the POD
basis is built from training Reynolds numbers only, so held-out values never
enter the representation. Strouhal number is taken from a wake probe and tracks
the Williamson correlation<sup>14</sup> with mean offset +0.09% and spread
0.90%.

**Navier–Stokes system.** The two-dimensional vorticity equation
∂w/∂t + u·∇w = νΔw + f, ∇·u = 0, w = ∇ × u is solved pseudo-spectrally on
[0, 1]² at 128 × 128 with the forcing of the Fourier neural operator
benchmark<sup>10</sup>, using an exponential-integrator Runge–Kutta scheme with
dealiasing. Viscosity was swept over 26 values from 10⁻⁴ to 1.5 × 10⁻² to locate
a periodic window: the flow is steady for ν ≥ 1.9 × 10⁻³, aperiodic for
ν ≤ 1.8 × 10⁻³, and periodic at ν = 1.7 × 10⁻³, where the record runs 51 periods
with enstrophy steady to 8 × 10⁻⁴ and a split-half instability of 4.8 × 10⁻⁹.
A 100-period rollout at this setting is 4002 steps.

**Period estimator.** The period is estimated by a least-squares Fourier fit
with unknown fundamental, seeded from the periodogram peak so the estimator
cannot be handed the answer, and validated against synthetic signals of known
period to 10⁻¹⁰. An earlier estimator based on phase regression of the analytic
signal is unbiased for a pure sinusoid but biased at 1.2 × 10⁻⁵ for
harmonic-rich signals; it was replaced, and every result reported here uses the
replacement. Re-measurement changed no ordering and no conclusion. The
per-Reynolds DMD results were the one case where the old bias exceeded the
quantity being measured, and there the δ entries moved by 10% to a factor of
nine while |λ| was unchanged to eight decimals in every case, because the
eigenvalue is read off the fitted operator and never touches the estimator.

**Relative period error and horizon.** δ = (T_model − T_true)/T_true, with
T_true measured from the reference trajectory by the same estimator, and
T_model measured from the second half of a model rollout so that any initial
transient is excluded. A phase slip of Δθ = 2πδN after N periods produces
relative RMS error 2|sin(Δθ/2)|, which first reaches √2 at N = 1/(4δ); this
defines the usable horizon quoted throughout. Ground truth at arbitrary horizon
comes from phase-wrapping the reference trajectory, exact to 6 × 10⁻¹⁴ given
measured period jitter of 5 × 10⁻⁷. A rollout is recorded as diverged if it
leaves a bounded region or produces a non-finite value, and no δ is quoted for
it.

**Split-half well-posedness test.** A record is halved, the period is fitted
independently on each half, and the relative disagreement between the two fits
is reported as the split-half instability. A value below 10⁻⁷ indicates a
well-posed single-period fit. The threshold was calibrated on two diffusively
coupled Stuart–Landau oscillators swept across the locking transition at
coupling ε = 0–0.8 and observation windows of 200, 400, 800 and 1600 time
units. Over those 24 records, locked cases refit to within 9.6 × 10⁻⁹ in the
worst case and quasi-periodic cases never better than 1.1 × 10⁻⁶, so the two
populations are separated by a factor of 118 and a threshold anywhere between
them classifies every case correctly.

**Latent surrogates.** All latent models act on the leading 8 POD coefficients
and are trained by Adam at learning rate 3 × 10⁻³ with cosine-annealed decay,
batch 64, gradient-norm clipping at 10, for 4000 epochs on 25-step rollout
segments, in double precision. `ode-mlp` is a tanh MLP vector field of width 96
and depth 3 integrated by RK4; `ode-deep` is the same at width 192 and depth 4;
`ar-residual` and `ar-direct` apply a width-96, depth-3 tanh MLP as a discrete
map in increment and direct parameterisations respectively; `gru` is a GRU cell
of hidden width 96 with a linear read-out; `linear-sgd` is a single 8 × 8 matrix
fitted by the same optimiser and objective. Parameter counts are 20,264
(ode-mlp, ar-residual, ar-direct), 114,440 (ode-deep), 31,304 (gru) and 64
(linear-sgd).

**Neural operators.** The Fourier neural operator<sup>10</sup> has width 32,
depth 4, and retains 16 × 12 Fourier modes on the wake and 16 × 16 on the
Navier–Stokes field, with a lifting 1 × 1 convolution, four spectral-convolution
blocks each paired with a 1 × 1 bypass, and a two-layer GELU projection head
(1,581,730 parameters). It is trained by Adam at learning rate 10⁻³ with
cosine-annealed decay, batch 8, for 3000 epochs on rollout segments of 8, 25 or 32 steps on the wake and 25 or
40 on the Navier–Stokes field, in direct and residual parameterisations, with
noise injection at amplitude 10⁻³ where stated. The 32- and 40-step objectives
are the shortest that span a full period on their respective systems, at 32.15
and 40.02 snapshots per period. Inputs are normalised by the maximum field
magnitude and a coordinate grid is appended. The DeepONet<sup>11</sup> follows
the standard branch–trunk construction with p = 128 basis functions, branch and
trunk widths 256, trunk depth 4, 16 Fourier features on the trunk, and the same
optimiser, batch size, epoch count and segment length. Learned models are run at
five seeds on the oscillator and wake and three on the Navier–Stokes field;
results are medians, with seed spread reported as the ratio of largest to
smallest |δ| across seeds. Where a configuration has diverging seeds, the median
training loss is taken over the surviving seeds.

**Regularisers.** The energy penalty, the Lipschitz penalty and training-time
noise injection are applied at two weights each (10⁻² and 10⁻¹ for the energy
penalty, 10⁻³ and 10⁻² for the other two) on the oscillator system at five seeds
per condition, against an unregularised baseline at δ = 6.0 × 10⁻⁴. The
transverse multiplier |λ₂| is computed from the monodromy matrix of the learned
vector field, obtained by integrating the variational equation around the
converged cycle.

**Dynamic mode decomposition.** Exact DMD<sup>12</sup> fits a one-step linear
map to the POD coefficients by least squares, and |λ| is the modulus of its
dominant eigenvalue. Optimized DMD<sup>13</sup> fits the same exponential model
against the entire trajectory by Levenberg–Marquardt on the exponents with the
linear coefficients projected out, in a free form with p independent (σ, ω)
pairs and a harmonic-locked form constraining the modes to harmonics of one
fundamental at σ = 0. Both are evaluated at 8 and 16 modes on a single-Reynolds
basis and on a basis pooled over 6 Reynolds numbers, and reported both
analytically, from the fitted exponents, and from a 100-period rollout of the
reconstruction. The two disagree only for the free form, which is the point of
reporting both.

**Parametric reduced-order models.** Operators are fitted independently at 13
Reynolds numbers (6 training, 5 interpolated, 2 extrapolated) and combined by
entrywise interpolation, mode-matched eigen-interpolation, generic stabilisation
clipping non-dominant |λ| to ≤ 1, neutrality imposed on the dominant pair, and
neutrality plus an imposed measured frequency. Stability counts and median |δ|
are reported over the 11 non-extrapolated operating points. The frequency
interpolant is fitted to measured frequencies at training Reynolds numbers only;
two interpolants were run, a quartic polynomial and one following the Williamson
correlation<sup>14</sup>.

**Statistics.** Correlations are Spearman rank correlations on log-transformed
quantities, two-sided and uncorrected for multiple comparisons.
Configuration-level correlations take the median over seeds within a
configuration as one observation, since seeds within a configuration are not
independent; the pooled and configuration-level coefficients are both reported
wherever they differ, because pooling heterogeneous model classes manufactures a
trend. Central tendencies are medians and dispersion is reported as a ratio of
extremes rather than a variance, because the distributions span orders of
magnitude. No data were excluded except where stated: predictions below 0.05 are
excluded from the observed/predicted ratios of Extended Data Table 8, because
near a recurrence the prediction is a difference of large phases and becomes
acutely sensitive to δ.

**Retracted measurement.** A momentum-exchange drag estimate used in earlier
work on this solver is retracted: it gives C_d = 1.05, 0.43, 0.04 and −0.20 over
D = 16, 24, 32 and 40, and negative drag is impossible. No result reported here
depends on it; Strouhal number comes from a wake probe and is unaffected.

**Compute.** Simulations and training were run on the Rochester Institute of
Technology SPORC cluster under Slurm, one seed per array task. Submission
scripts for every experiment are in the repository under `cluster/`.

## Data availability

The simulation outputs, trained-model summaries and per-run result records that
support the findings of this study are available in the repository cited under
Code availability, in `results/` as approximately 280 JSON records, one per run.
Raw lattice-Boltzmann and Navier–Stokes snapshot fields are regenerable from the
solver scripts in that repository and are not deposited because of their size;
they are available from the author on request.

## Code availability

All code used in this study — solvers, surrogate training, the period estimator,
the split-half test, cluster submission scripts and the figure-generating
script — is available at https://github.com/[user]/limit-cycle-neural-surrogates
under the MIT licence. The repository is private at the time of submission. Read-only access for
editors and referees is available on request for the duration of review, and the
repository will be made public and archived with a Zenodo DOI on acceptance.

## References

1. Samanta, S. & Behera, S. Autoregressive rollout error in latent-space
   reduced-order models of bluff-body wakes is accumulated phase drift. Preprint
   at https://arxiv.org/abs/2608.07189 (2026).
2. Li, Z. & Matzakos, N. M. Long-time trajectory approximation via SA-NODEs:
   model predictive and Floquet strategies. Preprint at
   https://arxiv.org/abs/2608.10738 (2026).
3. Brandstetter, J., Worrall, D. & Welling, M. Message passing neural PDE
   solvers. In *Int. Conf. Learning Representations* (2022).
4. McCabe, M., Harrington, P., Subramanian, S. & Brown, J. Towards stability of
   autoregressive neural operators. *Trans. Mach. Learn. Res.* (2023).
5. Nie, F. & Suzuki, Y. Jacobian-adaptive weighting for stability: enhancing
   long-term rollout of neural partial differential equation solvers via
   spatially-adaptive regularization. Preprint at
   https://arxiv.org/abs/2603.05538 (2026).
6. Lippe, P., Veeling, B. S., Perdikaris, P., Turner, R. E. & Brandstetter, J.
   PDE-Refiner: achieving accurate long rollouts with neural PDE solvers. *Adv.
   Neural Inf. Process. Syst.* **36** (2023).
7. Brunton, S. L., Budišić, M., Kaiser, E. & Kutz, J. N. Modern Koopman theory
   for dynamical systems. *SIAM Rev.* **64**, 229–340 (2022).
8. Amsallem, D. & Farhat, C. Interpolation method for adapting reduced-order
   models and application to aeroelasticity. *AIAA J.* **46**, 1803–1813 (2008).
9. Geuss, M., Panzer, H. & Lohmann, B. STABLE — a stability algorithm for
   parametric model reduction by matrix interpolation. *Math. Comput. Model.
   Dyn. Syst.* **22**, 307–322 (2016).
10. Li, Z. et al. Fourier neural operator for parametric partial differential
    equations. In *Int. Conf. Learning Representations* (2021).
11. Lu, L., Jin, P., Pang, G., Zhang, Z. & Karniadakis, G. E. Learning nonlinear
    operators via DeepONet based on the universal approximation theorem of
    operators. *Nat. Mach. Intell.* **3**, 218–229 (2021).
12. Schmid, P. J. Dynamic mode decomposition of numerical and experimental data.
    *J. Fluid Mech.* **656**, 5–28 (2010).
13. Askham, T. & Kutz, J. N. Variable projection methods for an optimized
    dynamic mode decomposition. *SIAM J. Appl. Dyn. Syst.* **17**, 380–416
    (2018).
14. Williamson, C. H. K. Oblique and parallel modes of vortex shedding in the
    wake of a circular cylinder at low Reynolds numbers. *J. Fluid Mech.* **206**,
    579–627 (1989).

## Acknowledgements

Computational resources were provided by the Rochester Institute of Technology
Research Computing SPORC cluster.

## Author contributions

J.K.S. is the sole author and conceived the study, wrote the code, ran the
experiments, analysed the results and wrote the manuscript.

## Competing interests

The author declares no competing interests.

## Figure legends

**Fig. 1 | Phase drift is fully accounted for by the period error.** Observed
phase-drift rate |dΔθ/dt| against the rate 2π|δ|/T predicted from the measured
relative period error, for 75 neural-ODE surrogates of the Stuart–Landau
oscillator spanning four decades of δ. Marker shape carries the sign of δ
(circles, δ > 0, n = 10; triangles, δ < 0, n = 65); magnitudes are plotted on
logarithmic axes because on linear axes all but the few worst models collapse
onto the origin. The dashed line is y = x, not a fit. Inset: histogram of the
observed/predicted ratio, median 1.0000 over the 75 models, range 1.000–1.033.

**Fig. 2 | What the period error costs.** Median relative error against rollout
length N in shedding periods, for five wake surrogates rolled out to 30,000
periods. **a**, Phase-resolved error, the relative RMS error at a specified instant.
Faint dashed curves are the law 2|sin(πδN)| evaluated at each model's own
measured δ, not a fit; the horizontal dashed line is √2, the decorrelated level,
first reached at N = 1/(4δ). The non-monotonic behaviour of `ode-mlp` past
10⁴ periods is the recurrence of the sine, not noise. **b**, Phase-invariant
error, the error in the covariance standing in for Reynolds stresses, which is
flat to within 1.1× over a 3000-fold increase in rollout length for every model
with an attractor. The two linear surrogates are the exception and the argument:
lacking an attractor, nothing bounds their amplitude but |λ| itself, and exact
DMD's |λ| = 1.00001312 compounds over 9.7 × 10⁵ steps.

**Fig. 3 | Loss orders model classes weakly, and models not at all.** Median
training MSE against median |δ| for 15 surrogate configurations spanning latent
neural ODEs, autoregressive maps, a GRU, a linear map, Fourier neural operators
on wake fields and on Navier–Stokes vorticity, and DeepONet. Faint markers are
individual seeds; solid markers are configuration medians; vertical bars span
each configuration's seed range. Colour carries model family and marker shape separates configurations within a
family; the palette is capped at five separable hues, so the Navier–Stokes
operators share the operator colour and are distinguished by marker. The bars are the argument: the spread
within one configuration at fixed data and objective, 1.9× to 6.0 × 10⁴×, is
taller than the trend the medians follow. Both rank correlations are annotated,
over the 12 wake configurations and over all 15.

**Fig. 4 | Two spectral properties, imposed one at a time.** Eigenvalues of a parametric reduced-order operator at all 13 Reynolds numbers,
plotted in the complex plane against the unit circle (dashed). All four panels
share one axis range, so points outside the frame are counted in the panel title
rather than dropped. The δ and horizon printed under each panel are medians over
the interpolated operating points; the train/interpolated split is in Extended
Data Table 6. Entrywise interpolation leaves the neutrally stable set
immediately, reaching spectral radius 31.05, and every rollout diverges. Generic
stabilisation clips every non-dominant eigenvalue with a positive growth rate
but cannot identify which one carries the phase, so it leaves the dominant pair
untouched at radius 2.96. Imposing neutrality on the dominant pair brings the
radius to exactly 1 and leaves a 3-period horizon. Imposing the measured
frequency as well leaves the spectrum visually unchanged while cutting δ from
7.6 × 10⁻² to 3.6 × 10⁻⁴: the last two panels differ in the angle of one
eigenvalue pair and in no radius, which is why the separating quantity is
printed above them.

**Fig. 5 | Conditioning on a parameter costs the horizon.** Usable horizon
1/(4|δ|) in shedding periods against Reynolds number for a free parametric
operator and two phase-anchored variants differing only in their frequency
interpolant. Shaded bands mark training Reynolds numbers; unshaded values between
them are interpolation and the two rightmost are extrapolation. The free operator falls from ~10⁴ periods at a single operating point to 433 at
interpolated Re and 9 outside the training range, while median training loss
varies by only 1.5× across the models compared. Anchoring wins
in-distribution and loses outside it, and both anchored variants fail together
despite predicted extrapolation errors differing by 4×, which is why the failure
is attributed to the decoder rather than to the frequency.

**Fig. 6 | δ is undefined at every setting the benchmark ships.** Split-half
instability of the period fit against viscosity for the two-dimensional
Navier–Stokes system in the forcing geometry of the Fourier neural operator
benchmark. Marker shape and colour give the regime classified independently of
the fit. Grey vertical bands mark two of the three viscosities the benchmark
ships, ν = 10⁻³ and 10⁻⁴; the third, ν = 10⁻⁵, lies to the left of the plotted
range and is likewise aperiodic. The dashed line at 10⁻⁷ is the well-posedness
threshold, below which a single-period fit means something. Only ν = 1.7 × 10⁻³
lies below it, at 4.8 × 10⁻⁹; the nearest aperiodic case, at ν = 1.6 × 10⁻³ on
the edge of the locking window, sits at 1.3 × 10⁻⁷, a factor of 1.3 above the
threshold and 27 above the periodic case.

## Extended Data

**Extended Data Table 1 | Three stability methods, measured on the phase
direction.** Neural ODEs on the Stuart–Landau oscillator, five seeds, relative to
an unregularised baseline at δ = 6.0 × 10⁻⁴. The δ ratio is median |δ| divided by
the baseline's; |λ₂| is the transverse Floquet multiplier of the learned vector
field. No method improves δ beyond seed noise, and every method moves |λ₂|.

| method | weight | δ ratio | transverse \|λ₂\| |
|---|---|---|---|
| none | — | 1.00 | 5.7 × 10⁻³ |
| energy penalty | 10⁻² | 0.91 | 5.7 × 10⁻³ |
| energy penalty | 10⁻¹ | 0.87 | 5.8 × 10⁻³ |
| Lipschitz | 10⁻³ | 1.87 | 1.1 × 10⁻² |
| Lipschitz | 10⁻² | 27.4 | 9.8 × 10⁻³ |
| noise injection | 10⁻³ | 1.08 | 5.5 × 10⁻³ |
| noise injection | 10⁻² | 2.73 | 4.5 × 10⁻³ |

**Extended Data Table 2 | Least-squares DMD recovers the neutral multiplier with
no training involved.** One-step linear maps fitted to the leading 8 POD
coefficients of the cylinder wake by exact DMD. |λ| is the modulus of the
dominant multiplier; δ is the relative period error of the fitted operator.

| Re | \|λ\| | δ |
|---|---|---|
| 80 | 1.00000000 | +2.52 × 10⁻⁸ |
| 100 | 1.00000000 | −2.82 × 10⁻⁸ |
| 130 | 1.00000000 | −2.14 × 10⁻⁷ |
| 180 | 0.99999995 | −6.57 × 10⁻⁶ |

**Extended Data Table 3 | Seven surrogate families on identical data, basis and
estimator.** Cylinder wake at Re = 100, 8-mode POD basis, five seeds each.
Harmonic-locked optimized DMD is carried at the top for reference. Seed spread is
the ratio of largest to smallest |δ| across seeds; horizon is 1/(4|δ|) in
shedding periods. δ spans six orders of magnitude; the loss column does not
order it.

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

**Extended Data Table 4 | Fourier neural operator on wake fields: the objective
sets the horizon.** FNO trained on u(x, y, t) → u(x, y, t + Δt) over the full 2 × 192 × 96 grid,
five seeds except seven for the 32-step row, rolled out 3215 steps (100 periods)
and projected onto the same 8-mode POD basis. "stable" counts seeds completing
the rollout, and median training MSE is taken over the surviving seeds. T is the
shedding period, 32.15 snapshots. The 8-step objective reaches the lowest
training loss of the four residual configurations and is the only one that
cannot finish a rollout; extending the objective until it closes the orbit gives
the best residual δ and the only configuration in which every seed survives.

| training objective | variant | stable | med train MSE | med \|δ\| | seed spread | horizon |
|---|---|---|---|---|---|---|
| 8-step rollout (0.25 T) | residual | 0/1 | **2.88 × 10⁻⁶** | diverged at step 645 | — | **0 P** |
| 25-step rollout (0.78 T) | residual | 4/5 | 7.3 × 10⁻⁶ | 4.48 × 10⁻⁴ | 141× | 558 P |
| **32-step rollout (1.00 T)** | residual | **7/7** | 9.79 × 10⁻⁶ | **1.93 × 10⁻⁴** | 102× | **1292 P** |
| 25-step rollout (0.78 T) | direct | 5/5 | 1.58 × 10⁻⁵ | **8.61 × 10⁻⁶** | **10×** | **29033 P** |
| 25-step + noise injection | residual | 5/5 | 1.92 × 10⁻⁵ | 4.90 × 10⁻⁴ | 890× | 510 P |

**Extended Data Table 5 | Fourier neural operator on the Navier–Stokes field at
ν = 1.7 × 10⁻³.** Operator acting on the full 128 × 128 vorticity field in the forcing geometry of
the FNO benchmark, three seeds, rollouts of 4002 steps (100 periods). T is the
period, 40.02 snapshots. At a 25-step objective the residual parameterisation
fits fifteen times better than the direct one and has a fifty times worse
period; extending that objective to close the orbit improves its δ by 81× for
1.8× in fit, which is the largest effect of any intervention in this study.

| training objective | variant | stable | med train MSE | med \|δ\| | seed spread | horizon |
|---|---|---|---|---|---|---|
| 25-step (0.62 T) | residual | 3/3 | **2.14 × 10⁻⁶** | 1.20 × 10⁻² | 12× | 21 P |
| **40-step (1.00 T)** | residual | 3/3 | 3.81 × 10⁻⁶ | **1.47 × 10⁻⁴** | **4.8×** | **1696 P** |
| 25-step (0.62 T) | direct | 3/3 | 3.24 × 10⁻⁵ | 2.40 × 10⁻⁴ | **2×** | 1042 P |
| 25-step + noise | residual | 3/3 | 9.69 × 10⁻⁶ | 1.89 × 10⁻³ | 148× | 132 P |

**Extended Data Table 6 | Repairing an interpolated parametric operator one
spectral property at a time.** Parametric reduced-order models across 13 Reynolds
numbers; columns give median |δ| at training and interpolated Reynolds numbers
and the count of stable rollouts. The last two rows agree to within 0.03%:
imposing both properties makes δ equal to the precision of the frequency
estimate and nothing else.

| construction | train | interp | stable |
|---|---|---|---|
| entry interpolation | diverges | diverges | 0/13 |
| mode-matched eigen-interpolation | 3.95 × 10⁻³ | 2.53 × 10⁻³ | 4/11 |
| generic stabilisation (clip non-dominant \|λ\| ≤ 1) | 4.07 × 10⁻¹ | 3.91 × 10⁻¹ | 6/11 |
| + neutrality on the dominant pair | 2.83 × 10⁻² | 7.64 × 10⁻² | 11/11 |
| **+ imposed measured frequency** | **4.50 × 10⁻⁴** | **3.65 × 10⁻⁴** | 11/11 |
| *frequency interpolant's own error* | *4.50 × 10⁻⁴* | *3.66 × 10⁻⁴* | — |

**Extended Data Table 7 | Conditioning a wake operator on Reynolds number.**
Median |δ| at training, interpolated and extrapolated Reynolds numbers, over
four seeds. The POD basis is built from training Reynolds numbers only. Median
training loss varies by 1.5× across the three models (5.8 × 10⁻⁶ to 8.5 × 10⁻⁶)
while δ spans three orders of magnitude; seed spread at fixed Reynolds number
runs 1.5–16× for the free operator.

| model | params | train | interp | extrap |
|---|---|---|---|---|
| free | 37392 | 7.33 × 10⁻⁴ | 7.14 × 10⁻⁴ | 3.38 × 10⁻² |
| free | 176272 | 3.88 × 10⁻⁴ | 5.78 × 10⁻⁴ | 2.68 × 10⁻² |
| anchored (poly ω) | 108080 | 2.55 × 10⁻⁵ | 3.17 × 10⁻⁴ | 1.79 × 10⁻¹ |

**Extended Data Table 8 | Phase-resolved error follows 2|sin(πδN)|.** Ratio of
observed to predicted relative RMS error at a specified instant, over rollouts
from 10 to 30,000 shedding periods. Predictions below 0.05 are excluded, because
near a recurrence the prediction is a difference of large phases and becomes
acutely sensitive to δ, which spans 1.5–4.8 × 10⁻⁵ across seeds here.

| model | median | p10 | p90 |
|---|---|---|---|
| ode-mlp | 1.036 | 0.969 | 1.103 |
| ar-residual | 1.023 | 0.966 | 1.112 |
| ar-direct | 1.034 | 0.988 | 1.699 |
| ode-deep | 1.080 | 0.705 | 1.135 |

**Extended Data Table 9 | Phase-invariant statistics stay flat, provided the
model has an attractor.** Error in the covariance standing in for Reynolds
stresses, at 10 periods and at 30,000 periods, a 3000-fold increase in rollout
length.

| model | N = 10 | N = 30000 | growth | attractor |
|---|---|---|---|---|
| ode-mlp | 6.2 × 10⁻³ | 6.7 × 10⁻³ | **1.1×** | nonlinear |
| ode-deep | 6.1 × 10⁻³ | 6.5 × 10⁻³ | **1.1×** | nonlinear |
| ar-residual | 7.3 × 10⁻³ | 7.8 × 10⁻³ | **1.1×** | nonlinear |
| ar-direct | 3.9 × 10⁻² | 4.0 × 10⁻² | **1.0×** | nonlinear |
| linear-sgd | 4.2 × 10⁻² | 9.9 × 10⁻¹ | 23.7× | linear |
| exact DMD | 3.5 × 10⁻² | 3.8 × 10⁹ | **1.1 × 10¹¹×** | linear |
| gru | 1.9 × 10⁻¹ | 1.1 × 10¹⁰ | 5.8 × 10¹⁰× | no cycle |

**Extended Data Table 10 | The boundary is one of well-posedness.** Two
Stuart–Landau oscillators with diffusive coupling ε, swept across the locking
transition. Half-vs-half instability is the relative disagreement between period fits on the
two halves of one record, quoted here at a record length of 400 time units. The
quasi-periodic entries are window-sensitive — at ε = 0 the same diagnostic reads
5.4 × 10⁻⁶, 3.8 × 10⁻¹, 1.1 × 10⁻⁵ and 1.3 × 10⁻⁵ at record lengths of 200, 400,
800 and 1600 — which is why the classification uses a fixed threshold rather
than the size of the disagreement. Below the locking transition the
single-period fit disagrees with itself and the δ it reports measures nothing,
while both true frequencies are recovered to 10⁻³; once locked, the fit is stable
to 10⁻⁹ and its δ agrees with a direct measurement.

| ε | regime | half-vs-half instability | δ from the fit | δ measured properly |
|---|---|---|---|---|
| 0.00 | quasi-periodic | **3.8 × 10⁻¹** | 6.08 × 10⁻¹ | 1.9 × 10⁻³, 1.8 × 10⁻³ |
| 0.10 | quasi-periodic | 1.3 × 10⁻⁴ | 6.11 × 10⁻¹ | 5.4 × 10⁻³, 4.6 × 10⁻³ |
| 0.30 | quasi-periodic | 1.5 × 10⁻⁵ | 1.16 × 10⁻⁴ | 1.0 × 10⁻⁴, 1.2 × 10⁻⁴ |
| 0.50 | locked | **1.8 × 10⁻⁹** | 2.84 × 10⁻⁴ | the same number |
| 0.80 | locked | **9.6 × 10⁻⁹** | 8.80 × 10⁻⁴ | the same number |
