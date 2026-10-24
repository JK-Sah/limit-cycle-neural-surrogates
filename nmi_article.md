# Long-horizon fidelity of learned surrogates of oscillatory systems is set by their Floquet spectrum

J. K. Sah

Department of Mechanical and Industrial Engineering, Rochester Institute of
Technology, Rochester, NY, USA. e-mail: js8472@rit.edu

---

## Abstract

A stable limit cycle has one Floquet multiplier at unity, forced by
time-translation invariance, and the rest inside the unit circle. A surrogate
with such a cycle corrects amplitude error but never phase error, so its
long-horizon fidelity reduces to one scalar, the relative period error δ:
measured over 100 periods it forecasts phase-resolved error 300 times further
out, while time-averaged statistics stay flat. Standard stability methods act on
the transverse multipliers instead, and a Lipschitz penalty makes δ 27 times
worse. Field operators add a precondition. Of 120 residual Fourier neural
operators trained on a cylinder wake, Navier–Stokes flow and a cardiac spiral
wave, 7 kept a clean limit cycle over 400 periods; their transverse multipliers
sit at 0.97 per period. Predicting the next state instead of the increment moved
them to 0.02 and kept the cycle in 28 of 30, with δ up to 1,900 times smaller.

---

Many of the systems learned surrogates are now built for oscillate: vortex wakes
and combustion instabilities, gaits and heartbeats, circadian and power-grid
rhythms. A surrogate of such a system is useful only for as long as it stays in
step with it, and nothing in standard training or validation measures how long
that is.

Rollout error in neural-PDE surrogates grows and then flattens near a plateau,
usually read as saturation of a generic accumulation process and treated with
generic remedies: longer training rollouts, noise injection, Jacobian penalties.
For bluff-body wakes the error is not generic. Samanta and Behera<sup>1</sup>
find that the autoregressive rollout error of a convolutional-autoencoder LSTM
at Re = 100–800 is 95–98% pure phase error, while amplitude is reproduced to
0.15% and one-step validation error looks perfect. We take that as established.
What it leaves open is why the error takes that form, whether it is specific to
learned models, what sets its size and which interventions reach it.

For a system on a limit cycle the answer is available in closed form. The
monodromy matrix of a periodic orbit of an autonomous system has one eigenvalue
exactly 1, for displacement along the orbit, by time-translation invariance,
learned or not; for a stable cycle the rest lie strictly inside the unit circle.
A transverse perturbation therefore decays and amplitude error self-corrects,
while a perturbation along the cycle neither grows nor decays and phase error is
never corrected. If a surrogate's period differs from the truth by a relative
amount δ, phase error grows as Δθ(t) ≈ −2πδt/T without bound, until the
prediction is antiphase and relative L² error saturates near √2: the plateau is
decorrelation, not saturation.

The argument predicts that long-horizon error is accounted for by one scalar
measurable from a short rollout, and that mechanisms acting on the transverse
directions cannot help. Both hold wherever the surrogate has a stable cycle.

Floquet theory has been brought to this problem before, with the opposite
prescription. Li and Matzakos<sup>2</sup> bound long-horizon neural-ODE error by
certifying that the learned return map contracts, and note that phase drift
between the two clocks persists. The bound is correct, but contraction acts on
transverse multipliers already inside the unit circle; we measure what that
costs. Samanta and Behera<sup>1</sup> remove the drift after training, with one
fitted rate per latent coordinate; we ask what sets it during training. How
error accumulates along a periodic orbit is a classical question for numerical
integrators<sup>3</sup>. What is new is that learned surrogates are trained,
selected and stabilised by criteria that never look at it. The rollout-stability
literature — pushforward and unrolled training<sup>4–6</sup>, noise
injection<sup>7</sup>, spectral-norm and Jacobian regularisation<sup>8,9</sup>,
spectral refinement<sup>10</sup>, residual and integration-constrained outputs
that hold the one-step Jacobian near the identity<sup>11–13</sup> — is evaluated
on rollout error, spectra or divergence rates, and work on chaotic systems has
turned to matching the invariant measure<sup>14–16</sup>. We are not aware of a
study that reports what any of it does to the frequency of a periodic solution.

Neutral eigenvalues are foundational in Koopman analysis<sup>17,18</sup>, where
unit modulus is the signature of sustained oscillation, and the phase is built
in by construction in phase-reduction theory<sup>19</sup>, in Koopman
autoencoders that learn their eigenvalue frequencies through an auxiliary
network<sup>20</sup> and in rhythmic movement primitives for
robots<sup>21</sup>. The contribution here is not that the neutral direction
exists but that, in a surrogate that does not build it in, its accuracy binds
the usable horizon, is measurable from a short rollout once the surrogate's
cycle attracts, and is absent from every loss and basis-selection criterion in
standard use. Parametric reduced-order models are likewise known to lose
stability under interpolation<sup>22,23</sup>; we sharpen the target by showing
generic stabilisation is insufficient for a limit cycle. Neural
operators<sup>24,25</sup> are tested because the claim is aimed at them, and
they expose a condition the argument takes for granted: that the surrogate has a
stable limit cycle at all. In the residual parameterisation standard for
autoregressive operators, most do not, and the reason is again a multiplier.

## Results

### The phase law is exact

We begin with the Stuart–Landau oscillator, where the cycle radius, angular
frequency and transverse multiplier are known in closed form (Methods). Numbers
are medians over five seeds and evaluation rollouts run 200 periods.

Over 75 trained models the ratio of observed phase drift to the rate predicted
from the learned period has median 1.0000 and range 1.000–1.033 (Fig. 1), with
median residual 1.4 × 10⁻³ rad from a straight-line fit to Δθ(t). Both
quantities come from the same learned cycle, so this is a consistency check
rather than a forecast: nothing but the period enters the drift. The forecast is
tested next. The computed leading multiplier deviates from 1 by at most 1.2 ×
10⁻¹⁰.

Phase is unbounded and amplitude is not. At 200 periods, median |Δθ| is 0.82 rad
and the worst case 52.0 rad, against a median maximum amplitude error of 3.3 ×
10⁻³ on a cycle of radius 0.707.

The training loss does not see any of this. Across the 40 runs of the scaling
sweep, spanning a 64× range of dataset size, log(train MSE) and log|δ| have ρ =
−0.21 (p = 0.19), and two runs with identical architecture and data and
indistinguishable loss (9.1 × 10⁻⁶ and 7.2 × 10⁻⁶) differ in δ by two orders of
magnitude, 1.4 × 10⁻³ against 1.1 × 10⁻⁵. Median |δ| does not fall with more
data past the smallest set: 1.6 × 10⁻³, 1.3 × 10⁻⁴, 2.6 × 10⁻⁴ and 3.3 × 10⁻⁴ at
16, 64, 256 and 1024 trajectories.

### What the period error predicts

The phase law makes a split prediction (Fig. 2). Phase-invariant quantities —
the time-mean field, RMS fluctuation, the covariance standing in for Reynolds
stresses — should not depend on where in the cycle a surrogate believes it is;
phase-resolved quantities should decorrelate at N = 1/(4δ). We test this on the
cylinder wake introduced below, rolling seven surrogate types, three seeds each,
to 30,000 periods with δ measured from the first 100. Phase-resolved error
follows 2|sin(πδN)| with per-model median observed/predicted ratios of 1.02–1.08
(Extended Data Table 1). In all ten runs of the four nonlinear surrogates whose
predicted horizon falls inside 30,000 periods, the error first reaches √2 at the
first sampled horizon beyond 1/(4δ), and the two runs predicted to outlast
30,000 periods do. A δ measured over 100 periods forecasts the error 300 times
further out. Covariance error over the same 3000-fold increase in rollout length
grows by 1.0–1.1× for every model with an attractor (Extended Data Table 2).

The exceptions are the argument rather than counterexamples. A linear surrogate
has no attractor, so nothing bounds its amplitude but |λ| itself, and exact
DMD's |λ| = 1.00001312 compounds over 9.7 × 10⁵ steps to a covariance error of
3.8 × 10⁹. Neutrality keeps amplitude bounded; without it even the
phase-invariant statistics fail.

### Stability methods act on the transverse direction

Relative to an unregularised baseline at δ = 6.0 × 10⁻⁴, none of three stability
methods in common use improves δ (one-sided permutation p ≥ 0.14 at every
weight, five seeds against five), and the Lipschitz penalty makes it worse at
both weights (p = 0.024), 27.4× at the stronger (Extended Data Table 3). The
transverse multiplier column is the point: these methods do act — the Lipschitz
penalty nearly doubles |λ₂|, from 5.7 × 10⁻³ to 1.1 × 10⁻² — and what they act
on is the direction that was already contracting.

### It is not a neural pathology

We next take flow past a cylinder at Re = 100, with latent dynamics on POD
coefficients (Methods). The period from POD coefficient 1 agrees with an
independent wake probe to 0.000%, confirming the latent phase is the physical
one.

Fitting a one-step linear map to those coefficients by least squares — dynamic
mode decomposition<sup>26</sup>, no network involved — recovers the neutral
multiplier to within 5 × 10⁻⁸ of unity at every Reynolds number tested (Extended
Data Table 4). The Floquet structure belongs to the problem, not to the
training.

Across seven surrogate families on the same 8-mode basis, data, estimator and
five seeds, δ spans six orders of magnitude and the loss column does not order
them usably (Extended Data Table 5). A GRU reaches lower training loss than
`linear-sgd` and `ar-direct`, which work, and has no usable limit cycle: its
period is wrong by a factor of forty. A recurrent hidden state carries dynamics
of its own and drifts over a long rollout, invisible to the objective.

The fitting objective matters within a fixed hypothesis class: DMD's linear map
fitted by Adam over 25-step rollouts rather than by least squares on one step is
6× better in δ, and optimized DMD<sup>27</sup>, fitted against the whole
trajectory, is 26,700× better. Training loss, by contrast, does not order these
surrogates by period error (Fig. 3). Taking each configuration as one
observation, the defensible unit since seeds within a configuration are not
independent, median training loss and median |δ| have ρ = +0.43 (p = 0.24)
across nine wake configurations and ρ = +0.02 (p = 0.95) across fourteen that
include the field operators below, while the spread within one configuration at
fixed data and objective runs from 2.1× to 1.4 × 10⁶×. Exact DMD's spread is
1.0× only because it is a deterministic closed-form fit.

### Field operators must first have a cycle

The models above act on eight POD coefficients, where a stable cycle comes
almost for free; neural operators, the models operator learning is built on, act
on fields. We train Fourier neural operators<sup>24</sup> on three fields: the
wake velocity on its full 2 × 192 × 96 grid; two-dimensional Navier–Stokes
vorticity at 128 × 128 in the forcing geometry of the FNO benchmark; and a
re-entrant spiral wave in the Aliev–Panfilov model of cardiac
tissue<sup>28</sup>, the standard model of re-entrant arrhythmia, as a 2 × 128 ×
128 field (Methods). Finding a usable Navier–Stokes case is part of the result:
at all three viscosities the benchmark ships, a split-half refit disagrees with
itself by 10⁻³ to 10⁻¹, so a δ quoted there would measure nothing, and a
viscosity sweep finds one periodic window, at ν = 1.7 × 10⁻³ (split-half 4.8 ×
10⁻⁹; Extended Data Fig. 1). The cardiac spiral is periodic to 9.0 × 10⁻⁹. Each
operator is rolled out 400 periods, and δ is fitted on periods 200–400 once its
own record has passed the split-half test applied to data.

In the residual parameterisation standard for autoregressive operators, which
predicts the increment, most surrogates never reach the regime the phase law
describes. Of 120 trained with segments of 0.25 to 2 periods, 26 diverge within
400 periods, 41 collapse onto a fixed point or a different rhythm, and 46 keep
an orbit whose period fails the split-half test; 7 keep a clean limit cycle
(Fig. 4b, Extended Data Table 6). The reason is in their own Floquet spectra.
With the phase multiplier at 1, the weakest transverse multiplier of 95 residual
operators has median 0.97 per period and none lies below 0.5 (Fig. 4a). Almost
nothing restores the amplitude, so slow drift carries the orbit off the cycle,
and a δ fitted over the first 100 periods, the protocol of earlier operator
studies including ours, describes a transient. Noise injection leaves the
multiplier where it was (median 0.96, nine models).

Predicting the next state instead of the increment changes this on every system.
The transverse multiplier falls to a median 0.019 per period and below 0.18 in
all 30 direct operators, and 28 of 30 keep a clean limit cycle at the true
period; of the other two, both on the cardiac field, one locks onto a clean
cycle at half the period and one onto a fixed point. At matched objective the
direct operator's δ is smaller in all six comparisons, by factors from 5 to over
1,000 (one-sided permutation p ≤ 0.048 in four, 0.08 and 0.11 in the other two),
and on the cardiac spiral it is 4.0 × 10⁻⁵ against 7.6 × 10⁻² (Fig. 4c). The
direct operators fit worse, by 1.4× to 25× in training loss. The
parameterisation that fits better is the one that loses the cycle, and the loss
does not report it.

Segment length matters less, and differently. A longer objective does not shrink
δ by any consistent rule. On Navier–Stokes a one-period objective is no better
than 0.62 of a period over five seeds, even on the 100-period fit of the earlier
protocol (median |δ| 3.1 × 10⁻³ against 4.4 × 10⁻⁴), and among residual runs
that keep a cycle, median δ moves without order across segment lengths on both
flows (from 8 × 10⁻⁶ to 1.3 × 10⁻⁴ on the wake). What a longer objective buys is
the cycle. Residual wake operators keep it in 3 of 10 runs below one period and
18 of 20 at or above (Fisher p = 0.002), and cardiac ones in 0 of 20 against 4
of 10 (p = 0.008, at 20% amplitude tolerance), with no detectable effect on
Navier–Stokes (Fig. 4d). Short objectives are where operators diverge: all five
wake operators at half a period, four of five Navier–Stokes ones at a quarter.

Where the cycle attracts strongly, δ is what the phase law says it is, and as
cheap to measure: for the 28 direct operators with a clean cycle, δ from periods
5–15 of the rollout matches δ from periods 200–400 to within 2.2% (median 0.0%).
For residual operators that keep an orbit, δ from the first ten periods misses
by a median 99%.

### Neutrality and frequency are both necessary

The clearest statement of the mechanism comes from parametric reduced-order
models. Operators fitted at a single Reynolds number have spectral radius 1 to
eight decimals. Interpolating between them entrywise, standard parametric-ROM
practice, gives radius 1.03 to 31.05: neutrally stable operators form a
measure-zero set, and linear interpolation leaves it immediately (Fig. 5).

Repairing the interpolated spectrum one property at a time across 13 Reynolds
numbers (Extended Data Table 7) shows that imposing both makes δ equal to the
precision of the frequency estimate and nothing else: the two final rows agree
to within 0.03%, 4.503 × 10⁻⁴ measured against 4.504 × 10⁻⁴ predicted by the
interpolant's own error. Because the rollout period is then 2π/ω by
construction, δ and the horizon are known before any model is fitted, from
measured frequencies alone.

Generic stabilisation, which clips every non-dominant eigenvalue with a positive
growth rate back onto the unit circle, cannot identify which carries the phase,
so it leaves the dominant pair untouched and the radius at 2.96. Neither
property suffices alone. Neutrality without the frequency stabilises every
rollout and leaves a 3-period horizon; frequency without neutrality diverges,
since correcting an imaginary part does nothing about a real part at 1.1 to 31.
Careful eigen-interpolation still diverges in 7 of 11 cases.

### Conditioning on a parameter costs the horizon

At a single operating point a well-trained free latent ODE reaches δ ≈ 2.6 ×
10⁻⁵ on the Re = 100 wake, a horizon near 10⁴ periods, past any engineering
need. But surrogates are built to cover a parameter range. Conditioning on
Reynolds number, with the POD basis built from training Re only, the free
operator holds 433 periods at interpolated Re and 9 outside the training range
against ~10⁴ at a single Reynolds number, while median training loss varies by
only 1.5× and seed spread stays at 1.5–16× (Fig. 6, Extended Data Table 8). The
same pattern appears on a parametric van der Pol oscillator: 71 periods at
held-out ε and 2 outside the range for the smallest of three model sizes, 265
and 16 for the largest. Conditioning costs roughly 20× in horizon, and the loss
does not report it.

Our own anchoring remedy fails here, for an identifiable reason. Anchoring the
phase to a measured frequency wins in-distribution, 15× at training parameters
and 1.8× at held-out ones, and loses outside the training range, at 1 period
against the free baseline's 9. Two frequency interpolants whose predicted
extrapolation errors differ by 4× produce measured errors coinciding to within
1–3% at Re = 200 and 220 while differing by up to 5× at interpolated Re, so ω is
not responsible. The observable is a = Z(φ, μ) + s, whose period equals 2π/ω(μ)
only when s decays and Z is a faithful loop, and both are learned functions of μ
that go out of distribution at unseen μ.

### Where the diagnostic stops

The analysis assumes a limit cycle: one neutral direction, one period, one
scalar. A quasi-periodic attractor has two incommensurate frequencies and no
single period, so the diagnostic cannot apply as stated. Rather than assert the
boundary we locate it, using two Stuart–Landau oscillators with diffusive
coupling swept across the locking transition (Extended Data Table 9). The
boundary is one of well-posedness: the single-period fit always latches onto the
stronger frequency, and what changes is whether the result means anything. At ε
= 0 the fit disagrees with itself by 38% on one record and its reported δ of
0.61 measures nothing, while both true frequencies are learned to 2 × 10⁻³. Once
locked, the fit is stable to 10⁻⁹ and its δ agrees with a direct measurement.

This yields a check worth running before any δ is quoted: split the record,
refit, compare. The separation is a threshold, not a ratio: over windows of 200
to 1600 time units, locked records refit to within 10⁻⁸ and quasi-periodic ones
never better than 1.1 × 10⁻⁶, while the margin between the worst of one and the
best of the other varies from 400× to 3,400× with the record. Applied to a
surrogate's own rollout, the same test is what separates the field operators
above. The obvious two-frequency generalisation of the error law does not hold
(Extended Data Table 9).

## Discussion

A learned surrogate of an oscillatory system keeps its long-horizon fidelity
only as far as its Floquet spectrum allows, and two parts of that spectrum
matter. The phase multiplier is fixed at one, so the accuracy of the frequency
it carries, δ, sets how long phase-resolved predictions stay correlated. The
transverse multipliers decide whether there is a stable cycle for δ to describe.
Standard practice measures neither. Training on field error constrains neither
directly: δ is orthogonal to an L² loss, and the residual parameterisation
leaves the transverse multipliers near one while fitting better than the
alternative. Interpolating operators does not preserve them, because neutrality
is measure-zero.

Why the residual parameterisation does this is not established here, but the
form of the map suggests an account. A residual step x + F(x) is the identity
plus a learned correction, so every multiplier starts at one and moves only
where the objective pushes it. That is why residual and integration-constrained
outputs are recommended for stability: they hold the one-step Jacobian near the
identity<sup>11,12</sup>, and on emulators of a chaotic system direct-step
models exceed the unit circle and diverge where integration-constrained ones do
not<sup>13</sup>; on turbulent flow residual targets make no measurable
difference<sup>29</sup>. Those studies measure stability as the absence of
divergence, and by that measure most residual operators here are stable: 94 of
120 never blow up, and most of those drift or collapse instead. On a limit
cycle, a Jacobian held near the identity in the transverse directions is what
removes the restoring force. A field operator's state has about 10⁴ dimensions
and its attractor a handful, so nearly every transverse direction is one the
training trajectories never probe; a direct map has no identity to fall back on
and must reconstruct the state, which contracts what lies off the training
manifold. The latent residual map on eight POD coefficients keeps its cycle
(Extended Data Tables 2, 5), as the account requires. Noise injection sits
uneasily with it: perturbing the inputs against clean targets should teach
contraction, yet left the multiplier at 0.96, although at 10⁻³ the perturbation
was under 1% of the field's fluctuation and larger amplitudes are untested here.

Methods that impose a neutral, correctly tuned cycle succeed whether learned or
classical, from harmonic-locked optimized DMD<sup>27</sup> to a repaired
parametric ROM, while our attempts to impose the phase on a learned model
directly fail out of distribution or fail to transfer between systems (Methods).
For operators what works is structural: a parameterisation whose cycle attracts.

The practical recommendations are short. Before quoting δ, check that both the
data and the surrogate's own long rollout are well-posed: split each record,
refit, compare. For a field operator, roll out hundreds of periods and compute
the leading transverse multiplier; near one, the cycle will not hold. Predict
the state rather than the increment, and train on segments of at least a period,
which keeps the cycle more often though it does little for δ. Do not read
long-horizon fidelity off a validation error, and report more than one seed. For
phase-resolved work the horizon is 1/(4δ); if the frequency can be measured,
impose it with neutrality, expecting the benefit in-distribution only.

The scope is narrower than the mechanism might suggest. Every system here has a
low-dimensional attractor, and every one the central results rest on is
periodic. Chaotic attractors have no periodic orbit and the argument does not
transfer; for the quasi-periodic side we locate the boundary and give a test,
but supply no replacement law. The parameterisation result rests on one operator
architecture at one width, three systems and five seeds per arm. The wake is
two-dimensional above Re ≈ 190, where the physical flow is not, so the
extrapolation points are numerically valid but physically artificial, and
parameter sweeps are one-dimensional. In the other direction the argument uses
nothing about fluids, only an autonomous model with a stable periodic attractor,
and the cardiac spiral shows it holding outside them; gaits, circadian rhythms
and grid oscillations are untested.

Our own results carry the paper's warning about single-seed δ: an earlier
three-seed comparison suggesting that a one-period objective improves δ 81-fold
on Navier–Stokes did not survive five seeds (Methods). The claims rest on the
spread at fixed everything, which no loss-based criterion can select against.

## Methods

**Oscillator system.** The Stuart–Landau oscillator is integrated in Cartesian
coordinates,

    ẋ = (μ − r²)x − (ω − βr²)y,   ẏ = (ω − βr²)x + (μ − r²)y,   r² = x² + y²,

with μ = 0.5, ω = 1.0 and β = −0.5. The cycle radius is √μ = 0.70711, the
angular frequency on the cycle is Ω = ω − βμ = 1.25, the period is T = 2π/Ω =
5.02655 and the transverse Floquet multiplier is exp(−2μT) = 6.55 × 10⁻³, all in
closed form. Reference trajectories are produced with an adaptive Dormand–Prince
(DOP853) integrator at rtol = atol = 10⁻¹². Training data are short segments
from randomised initial conditions on and near the cycle; the scaling sweep
spans 16, 64, 256 and 1024 trajectories.

**Wake system.** Flow past a circular cylinder is simulated with a D2Q9
lattice-Boltzmann solver, cylinder diameter D = 24 lattice units, inlet velocity
u_in = 0.1 in lattice units, kinematic viscosity ν = u_in D/Re and relaxation
time τ = 3ν + 0.5, over Re = 80–220. The domain is 80D × 20D with the cylinder
centred 20D from the inlet, giving 60D of downstream wake; this geometry was
chosen after domain truncation, not blockage or compressibility, was identified
as the source of an earlier Strouhal offset. Runs are 140,000 steps with
snapshots taken every 45 steps from the second half of the record, well past the
transient. Snapshots are cropped to a wake window running 2D upstream to 14D
downstream and ±4D across the centreline, then taken at stride 2, giving the 2 ×
192 × 96 velocity field the neural operators act on; the POD basis is built from
the same field. Fields are compressed by proper orthogonal decomposition using
the method of snapshots. Modes 1–2 form the shedding pair and carry 95.7% of the
energy at Re = 100. Latent surrogates act on the leading 8 POD coefficients
unless stated otherwise, at 32.15 snapshots per shedding period, so a 100-period
rollout is 3215 steps. For parametric studies the POD basis is built from
training Reynolds numbers only, so held-out values never enter the
representation. Strouhal number is taken from a wake probe and tracks the
Williamson correlation<sup>30</sup> with mean offset +0.09% and spread 0.90%.

**Navier–Stokes system.** The two-dimensional vorticity equation ∂w/∂t + u·∇w =
νΔw + f, ∇·u = 0, w = ∇ × u is solved pseudo-spectrally on [0, 1]² at 128 × 128
with the forcing of the Fourier neural operator benchmark<sup>24</sup>, using an
exponential-integrator Runge–Kutta scheme with dealiasing. Viscosity was swept
over 26 values from 10⁻⁴ to 1.5 × 10⁻² to locate a periodic window: the flow is
steady for ν ≥ 1.9 × 10⁻³, aperiodic for ν ≤ 1.8 × 10⁻³, and periodic at ν = 1.7
× 10⁻³, where the record runs 51 periods with enstrophy steady to 8 × 10⁻⁴ and a
split-half instability of 4.8 × 10⁻⁹. A 100-period rollout at this setting is
4002 steps. Eight POD modes carry 100.000% of the energy, so the attractor is
effectively eight-dimensional though the state has 16,384 components.

**Cardiac system.** The Aliev–Panfilov model of canine myocardium<sup>28</sup>,
∂u/∂t = ∇²u − ku(u − a)(u − 1) − uv, ∂v/∂t = ε(u, v)(−v − ku(u − a − 1)), ε = ε₀
+ μ₁v/(u + μ₂), is integrated with its published parameters k = 8, a = 0.15, ε₀
= 0.002, μ₁ = 0.2 and μ₂ = 0.3 by forward Euler at time step 0.02 with a
five-point Laplacian on a 256 × 256 grid of spacing 0.4 and no-flux boundaries.
A spiral is started by resetting the upper half of a planar wave to rest once
its front reaches 0.8 of the domain width; after a transient of 600 time units
the record runs 60 rotations, block-averaged 2 × 2 onto the 2 × 128 × 128 (u, v)
field the operators act on, at 39.7 snapshots per period. The rotation period is
33.38 time units (431 ms in the model's units) with split-half instability 9.0 ×
10⁻⁹; halving the time step moves it by 0.02% and refining the grid to spacing
0.32 by 0.29%. Eight POD modes carry 97.6% of the energy.

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
T_true measured from the reference trajectory by the same estimator, and T_model
measured from the second half of a model rollout so that any initial transient
is excluded. A phase slip of Δθ = 2πδN after N periods produces relative RMS
error 2|sin(Δθ/2)|, which first reaches √2 at N = 1/(4δ); this defines the
usable horizon quoted throughout. Ground truth at arbitrary horizon comes from
phase-wrapping the reference trajectory, exact to 6 × 10⁻¹⁴ given measured
period jitter of 5 × 10⁻⁷. A rollout is recorded as diverged if it leaves a
bounded region or produces a non-finite value, and no δ is quoted for it.

**Split-half well-posedness test.** A record is halved, the period is fitted
independently on each half, and the relative disagreement between the two fits
is reported as the split-half instability. A value below 10⁻⁷ indicates a
well-posed single-period fit. The threshold was calibrated on two diffusively
coupled Stuart–Landau oscillators swept across the locking transition at
coupling ε = 0–0.8 and observation windows of 200, 400, 800 and 1600 time units.
Over those 24 records, locked cases refit to within 9.6 × 10⁻⁹ in the worst case
and quasi-periodic cases never better than 1.1 × 10⁻⁶, so the two populations
are separated by a factor of 118 and a threshold anywhere between them
classifies every case correctly.

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

**Neural operators.** The Fourier neural operator<sup>24</sup> has width 32,
depth 4, and retains 16 × 12 Fourier modes on the wake and 16 × 16 on the
Navier–Stokes and cardiac fields, with a lifting 1 × 1 convolution, four
spectral-convolution blocks each paired with a 1 × 1 bypass, and a two-layer
GELU projection head (1,581,730 parameters on the wake, 2,105,857 on
Navier–Stokes, 2,106,018 on the cardiac field). In the residual parameterisation
it predicts the increment, x_{n+1} = x_n + F(x_n); in the direct one, the next
state, x_{n+1} = F(x_n). It is trained by Adam at learning rate 10⁻³ with
cosine-annealed decay over 3000 iterations, each on one batch of 8 segments of K
steps with start times drawn at random, so a K-step objective sees 8K
transitions per update; two controls hold 8K fixed while changing K (K = 25 at
batch 13 and K = 40 at batch 5 on Navier–Stokes). Inputs are normalised by the
maximum field magnitude and a coordinate grid is appended; noise injection,
where stated, adds Gaussian noise of standard deviation 10⁻³ to each input of
the unrolled loss. Every operator result comes from one protocol, run on NVIDIA
A100 GPUs in float64 with initialisation and segment order both seeded, five
seeds per configuration. After training each model is rolled out 400 periods
from the first training snapshot and projected onto the training POD basis; δ is
fitted on periods 200–400, its split-half instability on periods 200–300 against
300–400, and δ on the second half of the first 100 periods, the protocol of
earlier operator studies including ours, is recorded alongside. Gradient
checkpointing keeps long segments within 40 GB without changing the arithmetic.
Training under this protocol is deterministic: configurations repeated in two
separate sweeps reproduced training loss and δ to the printed precision. Model
weights are saved with the results. Latent models are run at five seeds on the
oscillator and the wake; results are medians, with seed spread reported as the
ratio of largest to smallest |δ|.

**Outcome of an operator rollout.** Each 400-period rollout is classified as
diverged (it left the bounded region within 100 periods), late-diverged (within
400), collapsed or locked onto a wrong rhythm (the RMS amplitude of the leading
POD pair over periods 390–400 fell below 80% of its value over periods 0–10, or
the late orbit no longer oscillates near the true period, |δ| > 0.5), retaining
a cycle, or retaining a well-posed cycle (late split-half instability below
10⁻⁷, the threshold applied to data). The second collapse test catches a model
that falls onto a fixed point within its first period and so keeps a constant
amplitude. The 20% tolerance was set after inspecting the rollouts; at 10% no
residual cardiac operator holds its amplitude, and at 50% the comparisons are
unchanged or stronger.

**Floquet multipliers of trained operators.** Each saved operator is rolled
forward 300 periods from the first training snapshot to settle onto its own
attractor. Four tangent vectors, the first along the orbit and the rest random,
are then carried forward by the operator's Jacobian and re-orthonormalised by QR
at every step, for 10 periods of burn-in and 30 accumulated periods.
Jacobian–vector products are forward differences with step 10⁻⁶ on unit vectors,
accurate to about 10⁻⁹ in float64. The resulting Lyapunov exponents λᵢ are
reported as multipliers per period exp(λᵢT): for a limit cycle the leading one
is 1, the phase direction, and the next is the weakest transverse contraction.
On a Stuart–Landau map embedded with three linearly decaying coordinates the
procedure recovers the exact multipliers per period, 1, 0.221 and 0.00656, to
the five digits printed.

**Earlier operator runs and an architecture that did not train.** Before this
protocol, operators were trained on CPUs with unseeded initialisation, two to
eight seeds per configuration, and δ fitted on periods 50–100. Those runs had
suggested that a one-period objective improves δ 81-fold on Navier–Stokes; the
improvement did not survive five-seed replication (median |δ| 4.4 × 10⁻⁴ at 25
steps against 3.1 × 10⁻³ at 40, on the same 100-period fit), and no result here
depends on them except the phase-penalty arms below. DeepONet runs under that
protocol are in the repository and are not reported, because their rollouts were
not saved for the well-posedness check. A convolutional U-Net<sup>31,32</sup>
(1,926,177 parameters, circular padding, residual parameterisation, sinusoidal
coordinate channels) was trained under the same protocol, with default and with
a zero-initialised output layer. Under either, half the seeds stayed on a loss
plateau near the field variance (training MSE ≈ 3 × 10⁻²) and the rest reached 3
× 10⁻⁵ to 3 × 10⁻³, against 2–4 × 10⁻⁶ for the FNO; none kept a well-posed cycle
(late split-half 3 × 10⁻³ or worse). With half its seeds untrained we draw no
architecture conclusion from it.

**Regularisers.** The energy penalty, the Lipschitz penalty and training-time
noise injection are applied at two weights each (10⁻² and 10⁻¹ for the energy
penalty, 10⁻³ and 10⁻² for the other two) on the oscillator system at five seeds
per condition, against an unregularised baseline at δ = 6.0 × 10⁻⁴. The
transverse multiplier |λ₂| is computed from the monodromy matrix of the learned
vector field, obtained by integrating the variational equation around the
converged cycle.

**Dynamic mode decomposition.** Exact DMD<sup>26</sup> fits a one-step linear
map to the POD coefficients by least squares, and |λ| is the modulus of its
dominant eigenvalue. Optimized DMD<sup>27</sup> fits the same exponential model
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
correlation<sup>30</sup>.

**Statistics.** Correlations are Spearman rank correlations on log-transformed
quantities, two-sided and uncorrected for multiple comparisons.
Configuration-level correlations take the median over seeds within a
configuration as one observation, since seeds within a configuration are not
independent; the pooled and configuration-level coefficients are both reported
wherever they differ, because pooling heterogeneous model classes manufactures a
trend (oscillator runs: ρ = +0.30, p = 0.009 pooled over all 75; ρ = −0.22, p =
0.15 within the 45 unregularised). Central tendencies are medians and dispersion
is reported as a ratio of extremes rather than a variance, because the
distributions span orders of magnitude. Two-arm comparisons of δ use an exact
one-sided permutation test on the difference in median log₁₀|δ|, enumerating
every relabelling of the pooled seeds; the same test is applied to remedies we
report as failed and to those we recommend. Fractions of seeds are compared by
one-sided Fisher exact tests. No data were excluded except where stated:
predictions below 0.05 are excluded from the observed/predicted ratios of
Extended Data Table 1, because near a recurrence the prediction is a difference
of large phases and becomes acutely sensitive to δ.

**Imposing the period in the objective.** Under the earlier CPU protocol, we
tested whether δ can be controlled during training rather than measured
afterwards, by adding to the rollout loss a Huber penalty on the relative period
error implied by each training segment, computed from the mean angular advance
of the whitened leading POD pair and scaled against the measured reference
period. A weight sweep on the wake at 10⁻², 10⁻¹, 1, 10 and 10³ located a narrow
usable band: at 10³ the penalty gradient exceeds the fit gradient by four orders
and training loss rises from 4.7 × 10⁻⁴ to 8.4 on every seed within 50 epochs,
while at 10 the resulting δ is 3.5× worse than doing nothing. At 10⁻¹, the best
weight, the wake median |δ| improves 2.8× (6.84 × 10⁻⁵ over eight seeds against
1.94 × 10⁻⁴ over seven), but an exact permutation test gives p = 0.18 and the
seed spread is unchanged, 109× against 102×, with Brown–Forsythe on log₁₀|δ|
giving p = 0.69. On the Navier–Stokes field the same weight destroys the model
on both seeds, training loss 3.6 × 10⁻² against the control's 3.8 × 10⁻⁶. The
usable weight is a function of the system's loss scale, so the method does not
transfer, and we report it as a failed remedy rather than a recommendation.

**Retracted measurement.** A momentum-exchange drag estimate used in earlier
work on this solver is retracted: it gives C_d = 1.05, 0.43, 0.04 and −0.20 over
D = 16, 24, 32 and 40, and negative drag is impossible. No result reported here
depends on it; Strouhal number comes from a wake probe and is unaffected.

**Compute.** Simulations and training were run on the Rochester Institute of
Technology SPORC cluster under Slurm, one seed per array task; field operators
on NVIDIA A100 GPUs, everything else on CPUs. Submission scripts for every
experiment are in the repository under `cluster/`.

**Use of AI tools.** Anthropic's Claude was utilized for proofreading and
language refinement, for assistance with code development, and for manuscript
preparation. All AI-assisted outputs were reviewed and verified by the author,
who takes full responsibility for the content.

## Data availability

The per-run result records that support the findings of this study, about 730
JSON files, and the projected 400-period rollouts of every field operator are in
the repository cited under Code availability, under `results/`. The trained
weights of the 170 field operators (4.9 GB) will be archived with the Zenodo
deposit at acceptance and are available from the author on request before then.
Raw lattice-Boltzmann, Navier–Stokes and cardiac snapshot fields are regenerable
from the solver scripts in the repository and are not deposited because of their
size; they are available from the author on request.

## Code availability

All code used in this study — solvers, surrogate training, the period estimator,
the split-half test, cluster submission scripts and the figure-generating script
— is publicly available at
https://github.com/JK-Sah/limit-cycle-neural-surrogates under the MIT licence.
The exact version used for this manuscript is tagged `nmi-submission` and is
also supplied to editors and referees as Supplementary Software; it will be
archived with a Zenodo DOI on acceptance.

## References

1. Samanta, S. & Behera, S. Autoregressive rollout error in latent-space
   reduced-order models of bluff-body wakes is accumulated phase drift. Preprint
   at https://arxiv.org/abs/2608.07189 (2026).
2. Li, Z. & Matzakos, N. M. Long-time trajectory approximation via SA-NODEs:
   model predictive and Floquet strategies. Preprint at
   https://arxiv.org/abs/2608.10738 (2026).
3. Cano, B. & Sanz-Serna, J. M. Error growth in the numerical integration of
   periodic orbits, with application to Hamiltonian and reversible systems.
   *SIAM J. Numer. Anal.* **34**, 1391–1417 (1997).
4. Brandstetter, J., Worrall, D. & Welling, M. Message passing neural PDE
   solvers. In *Int. Conf. Learning Representations* (2022).
5. Kochkov, D. et al. Machine learning–accelerated computational fluid dynamics.
   *Proc. Natl Acad. Sci. USA* **118**, e2101784118 (2021).
6. List, B., Chen, L.-W. & Thuerey, N. Learned turbulence modelling with
   differentiable fluid solvers: physics-based loss functions and optimisation
   horizons. *J. Fluid Mech.* **949**, A25 (2022).
7. Sanchez-Gonzalez, A. et al. Learning to simulate complex physics with graph
   networks. In *Proc. 37th Int. Conf. Machine Learning* 8459–8468 (PMLR, 2020).
8. McCabe, M., Harrington, P., Subramanian, S. & Brown, J. Towards stability of
   autoregressive neural operators. *Trans. Mach. Learn. Res.* (2023).
9. Nie, F. & Suzuki, Y. Jacobian-adaptive weighting for stability: enhancing
   long-term rollout of neural partial differential equation solvers via
   spatially-adaptive regularization. Preprint at
   https://arxiv.org/abs/2603.05538 (2026).
10. Lippe, P., Veeling, B. S., Perdikaris, P., Turner, R. E. & Brandstetter, J.
   PDE-Refiner: achieving accurate long rollouts with neural PDE solvers. *Adv.
   Neural Inf. Process. Syst.* **36** (2023).
11. Haber, E. & Ruthotto, L. Stable architectures for deep neural networks.
   *Inverse Probl.* **34**, 014004 (2018).
12. Hernández Noguera, E., Ferdaus, M. M., Ioup, E., Abdelguerfi, M. & Simeonov,
   J. Bridging spectral operator learning and U-Net hierarchies: SpectraNet for
   stable autoregressive PDE surrogates. Preprint at
   https://arxiv.org/abs/2605.09096 (2026).
13. Ainslie, C., Hassanzadeh, P., Mahoney, M. W. & Chattopadhyay, A.
   Eigenanalysis framework for autoregressive neural emulators of multi-scale
   chaotic dynamics. Preprint at https://arxiv.org/abs/2608.16084 (2026).
14. Jiang, R., Lu, P. Y., Orlova, E. & Willett, R. Training neural operators to
   preserve invariant measures of chaotic attractors. *Adv. Neural Inf. Process.
   Syst.* **36** (2023).
15. Schiff, Y. et al. DySLIM: dynamics stable learning by invariant measure for
   chaotic systems. In *Proc. 41st Int. Conf. Machine Learning* 43649–43684
   (PMLR, 2024).
16. Park, J., Yang, N. T. & Chandramoorthy, N. When are dynamical systems
   learned from time series data statistically accurate? *Adv. Neural Inf.
   Process. Syst.* **37** (2024).
17. Brunton, S. L., Budišić, M., Kaiser, E. & Kutz, J. N. Modern Koopman theory
   for dynamical systems. *SIAM Rev.* **64**, 229–340 (2022).
18. Mauroy, A. & Mezić, I. On the use of Fourier averages to compute the global
   isochrons of (quasi)periodic dynamics. *Chaos* **22**, 033112 (2012).
19. Nakao, H. Phase reduction approach to synchronisation of nonlinear
   oscillators. *Contemp. Phys.* **57**, 188–214 (2016).
20. Lusch, B., Kutz, J. N. & Brunton, S. L. Deep learning for universal linear
   embeddings of nonlinear dynamics. *Nat. Commun.* **9**, 4950 (2018).
21. Ijspeert, A. J., Nakanishi, J., Hoffmann, H., Pastor, P. & Schaal, S.
   Dynamical movement primitives: learning attractor models for motor behaviors.
   *Neural Comput.* **25**, 328–373 (2013).
22. Amsallem, D. & Farhat, C. Interpolation method for adapting reduced-order
   models and application to aeroelasticity. *AIAA J.* **46**, 1803–1813 (2008).
23. Geuss, M., Panzer, H. & Lohmann, B. STABLE — a stability algorithm for
   parametric model reduction by matrix interpolation. *Math. Comput. Model.
   Dyn. Syst.* **22**, 307–322 (2016).
24. Li, Z. et al. Fourier neural operator for parametric partial differential
    equations. In *Int. Conf. Learning Representations* (2021).
25. Lu, L., Jin, P., Pang, G., Zhang, Z. & Karniadakis, G. E. Learning nonlinear
    operators via DeepONet based on the universal approximation theorem of
    operators. *Nat. Mach. Intell.* **3**, 218–229 (2021).
26. Schmid, P. J. Dynamic mode decomposition of numerical and experimental data.
    *J. Fluid Mech.* **656**, 5–28 (2010).
27. Askham, T. & Kutz, J. N. Variable projection methods for an optimized
    dynamic mode decomposition. *SIAM J. Appl. Dyn. Syst.* **17**, 380–416
    (2018).
28. Aliev, R. R. & Panfilov, A. V. A simple two-variable model of cardiac
   excitation. *Chaos Solitons Fractals* **7**, 293–301 (1996).
29. Kohl, G., Chen, L.-W. & Thuerey, N. Benchmarking autoregressive conditional
   diffusion models for turbulent flow simulation. Preprint at
   https://arxiv.org/abs/2309.01745 (2023).
30. Williamson, C. H. K. Oblique and parallel modes of vortex shedding in the
    wake of a circular cylinder at low Reynolds numbers. *J. Fluid Mech.* **206**,
    579–627 (1989).
31. Ronneberger, O., Fischer, P. & Brox, T. U-Net: convolutional networks for
   biomedical image segmentation. In *Medical Image Computing and
   Computer-Assisted Intervention – MICCAI 2015* 234–241 (Springer, 2015).
32. Gupta, J. K. & Brandstetter, J. Towards multi-spatiotemporal-scale
   generalized PDE modeling. *Trans. Mach. Learn. Res.* (2023).

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
periods. **a**, Phase-resolved error, the relative RMS error at a specified
instant. Faint dashed curves are the law 2|sin(πδN)| evaluated at each model's
own measured δ, not a fit; the horizontal dashed line is √2, the decorrelated
level, first reached at N = 1/(4δ). The non-monotonic behaviour of `ode-mlp`
past 10⁴ periods is the recurrence of the sine, not noise. **b**,
Phase-invariant error, the error in the covariance standing in for Reynolds
stresses, which is flat to within 1.1× over a 3000-fold increase in rollout
length for every model with an attractor. The two linear surrogates are the
exception and the argument: lacking an attractor, nothing bounds their amplitude
but |λ| itself, and exact DMD's |λ| = 1.00001312 compounds over 9.7 × 10⁵ steps.

**Fig. 3 | Training loss does not order surrogates by period error.** Median
training MSE against median |δ| for 14 configurations: six latent families on
the cylinder wake (continuous and discrete neural maps, a GRU, a linear map),
and residual, noise-injected and direct Fourier neural operators on the wake,
Navier–Stokes and cardiac fields, whose δ is fitted on periods 200–400. Faint
markers are individual seeds; solid markers are configuration medians; vertical
bars span each configuration's seed range. Colour carries model family and
marker shape separates configurations within a family. Across configurations the
rank correlation is +0.43 (p = 0.24) on the nine wake configurations and +0.02
(p = 0.95) on all fourteen, and the spread within one configuration, 2.1× to 1.4
× 10⁶×, is taller than any trend. The direct operators sit to the right of the
residual ones, fitting worse, and below them.

**Fig. 4 | Residual operators lose the limit cycle; direct ones keep it.**
Fourier neural operators on three fields, five seeds per arm (Extended Data
Table 6). **a**, Weakest transverse Floquet multiplier per period of each
trained operator, computed from its own Jacobian along its attractor; the phase
multiplier is 1 throughout. Bars are medians. Residual operators, with or
without noise injection, sit at 0.97, where almost nothing restores the
amplitude; the direct operators' medians are 0.002 to 0.06 on the three systems
(one direct cardiac model, which fell onto a fixed point, lies off-scale below).
**b**, What 400 periods of rollout do, pooled over segment lengths: a well-posed
cycle passes the split-half test on periods 200–400. **c**, Late |δ| at matched
objectives; filled markers are seeds with a well-posed cycle, bars medians over
seeds that did not diverge (diverged seeds at the top). The two open direct
markers at cardiac K/T = 1.01 are a clean cycle at half the true period and a
fixed point. **d**, Fraction of residual seeds that keep a cycle against segment
length: a longer objective keeps the cycle more often on the wake and cardiac
fields, gradually and with no change at one period, and not detectably on
Navier–Stokes.

**Fig. 5 | Two spectral properties, imposed one at a time.** Eigenvalues of a
parametric reduced-order operator at all 13 Reynolds numbers, plotted in the
complex plane against the unit circle (dashed). All four panels share one axis
range, so points outside the frame are counted in the panel title rather than
dropped. The δ and horizon printed under each panel are medians over the
interpolated operating points; the train/interpolated split is in Extended Data
Table 7. Entrywise interpolation leaves the neutrally stable set immediately,
reaching spectral radius 31.05, and every rollout diverges. Generic
stabilisation clips every non-dominant eigenvalue with a positive growth rate
but cannot identify which one carries the phase, so it leaves the dominant pair
untouched at radius 2.96. Imposing neutrality on the dominant pair brings the
radius to exactly 1 and leaves a 3-period horizon. Imposing the measured
frequency as well leaves the spectrum visually unchanged while cutting δ from
7.6 × 10⁻² to 3.6 × 10⁻⁴: the last two panels differ in the angle of one
eigenvalue pair and in no radius, which is why the separating quantity is
printed above them.

**Fig. 6 | Conditioning on a parameter costs the horizon.** Usable horizon
1/(4|δ|) in shedding periods against Reynolds number for a free parametric
operator and two phase-anchored variants differing only in their frequency
interpolant. Shaded bands mark training Reynolds numbers; unshaded values
between them are interpolation and the two rightmost are extrapolation. The free
operator falls from ~10⁴ periods at a single operating point to 433 at
interpolated Re and 9 outside the training range, while median training loss
varies by only 1.5× across the models compared. Anchoring wins in-distribution
and loses outside it, and both anchored variants fail together despite predicted
extrapolation errors differing by 4×, which is why the failure is attributed to
the decoder rather than to the frequency.

## Extended Data

**Extended Data Fig. 1 | δ is undefined at every setting the benchmark ships.**
Split-half instability of the period fit against viscosity for the
two-dimensional Navier–Stokes system in the forcing geometry of the Fourier
neural operator benchmark. Marker shape and colour give the regime classified
independently of the fit. Grey vertical bands mark two of the three viscosities
the benchmark ships, ν = 10⁻³ and 10⁻⁴; the third, ν = 10⁻⁵, lies to the left of
the plotted range and is likewise aperiodic. The dashed line at 10⁻⁷ is the
well-posedness threshold, below which a single-period fit means something. Only
ν = 1.7 × 10⁻³ lies below it, at 4.8 × 10⁻⁹; the nearest aperiodic case, at ν =
1.6 × 10⁻³ on the edge of the locking window, sits at 1.3 × 10⁻⁷, a factor of
1.3 above the threshold and 27 above the periodic case.

**Extended Data Table 1 | Phase-resolved error follows 2|sin(πδN)|.** Ratio of
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

**Extended Data Table 2 | Phase-invariant statistics stay flat, provided the
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

**Extended Data Table 3 | Three stability methods, measured on the phase
direction.** Neural ODEs on the Stuart–Landau oscillator, five seeds, relative
to an unregularised baseline at δ = 6.0 × 10⁻⁴. The δ ratio is median |δ|
divided by the baseline's; |λ₂| is the transverse Floquet multiplier of the
learned vector field. No method improves δ beyond seed noise, and every method
moves |λ₂|.

| method | weight | δ ratio | transverse \|λ₂\| |
|---|---|---|---|
| none | — | 1.00 | 5.7 × 10⁻³ |
| energy penalty | 10⁻² | 0.91 | 5.7 × 10⁻³ |
| energy penalty | 10⁻¹ | 0.87 | 5.8 × 10⁻³ |
| Lipschitz | 10⁻³ | 1.87 | 1.1 × 10⁻² |
| Lipschitz | 10⁻² | 27.4 | 9.8 × 10⁻³ |
| noise injection | 10⁻³ | 1.08 | 5.5 × 10⁻³ |
| noise injection | 10⁻² | 2.73 | 4.5 × 10⁻³ |

**Extended Data Table 4 | Least-squares DMD recovers the neutral multiplier with
no training involved.** One-step linear maps fitted to the leading 8 POD
coefficients of the cylinder wake by exact DMD. |λ| is the modulus of the
dominant multiplier; δ is the relative period error of the fitted operator.

| Re | \|λ\| | δ |
|---|---|---|
| 80 | 1.00000000 | +2.52 × 10⁻⁸ |
| 100 | 1.00000000 | −2.82 × 10⁻⁸ |
| 130 | 1.00000000 | −2.14 × 10⁻⁷ |
| 180 | 0.99999995 | −6.57 × 10⁻⁶ |

**Extended Data Table 5 | Seven surrogate families on identical data, basis and
estimator.** Cylinder wake at Re = 100, 8-mode POD basis, five seeds each.
Harmonic-locked optimized DMD is carried at the top for reference. Seed spread
is the ratio of largest to smallest |δ| across seeds; horizon is 1/(4|δ|) in
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

**Extended Data Table 6 | Every field-operator arm, on one protocol.** Fourier
neural operators on the three fields, five seeds per row, A100 in float64 with
seeded initialisation, each rolled out 400 periods. K/T is the training segment
in periods and batch the segments per update. Outcomes per seed: diverged
(within 400 periods), collapsed onto a fixed point or a different rhythm
(amplitude of the leading POD pair below 80% of its starting value after 400
periods, or late |δ| > 0.5), an orbit whose late record fails the split-half
test, or a well-posed limit cycle (late split-half < 10⁻⁷). |δ| is fitted on
periods 200–400 and its median taken over seeds that keep a cycle. The
transverse multiplier is the median over seeds of the weakest transverse Floquet
multiplier per period of the trained model (Methods). Training MSE is the median
over all seeds. Residual operators fit as well as or better than direct ones and
keep a well-posed cycle in 7 of 120 runs; direct operators keep one in 28 of 30.

| system | parameterisation | K/T | batch | diverged | collapsed or wrong rhythm | cycle, not well-posed | well-posed cycle | median train MSE | median \|δ\| (runs keeping a cycle) | transverse multiplier |
|---|---|---|---|---|---|---|---|---|---|---|
| wake | residual | 0.50 | 8 | 5 | 0 | 0 | 0 | 4.85 × 10⁻⁶ | — | — |
| wake | residual | 0.75 | 8 | 1 | 1 | 3 | 0 | 6.41 × 10⁻⁶ | 1.18 × 10⁻⁴ | 0.97 |
| wake | residual | 1.00 | 8 | 1 | 0 | 4 | 0 | 1.07 × 10⁻⁵ | 6.93 × 10⁻⁵ | 0.97 |
| wake | residual | 1.24 | 8 | 1 | 0 | 4 | 0 | 1.35 × 10⁻⁵ | 2.96 × 10⁻⁵ | 0.97 |
| wake | residual | 1.49 | 8 | 0 | 0 | 5 | 0 | 1.92 × 10⁻⁵ | 1.32 × 10⁻⁴ | 0.96 |
| wake | residual | 1.99 | 8 | 0 | 0 | 4 | 1 | 2.60 × 10⁻⁵ | 8.43 × 10⁻⁶ | 0.96 |
| wake | residual + noise | 0.75 | 8 | 0 | 2 | 3 | 0 | 1.69 × 10⁻⁵ | 1.13 × 10⁻⁴ | 0.97 |
| wake | direct | 0.75 | 8 | 0 | 0 | 0 | 5 | 1.31 × 10⁻⁵ | 1.70 × 10⁻⁵ | 0.058 |
| wake | direct | 1.00 | 8 | 0 | 0 | 0 | 5 | 1.52 × 10⁻⁵ | 2.58 × 10⁻⁶ | 7.28 × 10⁻³ |
| Navier–Stokes | residual | 0.25 | 8 | 4 | 1 | 0 | 0 | 6.55 × 10⁻⁷ | — | 0.97 |
| Navier–Stokes | residual | 0.50 | 8 | 3 | 0 | 2 | 0 | 1.97 × 10⁻⁶ | 1.00 × 10⁻⁴ | 0.99 |
| Navier–Stokes | residual | 0.62 | 13 | 0 | 5 | 0 | 0 | 1.97 × 10⁻⁶ | — | 0.95 |
| Navier–Stokes | residual | 0.62 | 8 | 2 | 1 | 1 | 1 | 2.14 × 10⁻⁶ | 8.79 × 10⁻⁴ | 0.96 |
| Navier–Stokes | residual | 0.75 | 8 | 0 | 1 | 3 | 1 | 2.81 × 10⁻⁶ | 2.00 × 10⁻³ | 0.97 |
| Navier–Stokes | residual | 0.87 | 8 | 0 | 2 | 2 | 1 | 3.44 × 10⁻⁶ | 4.71 × 10⁻⁴ | 0.98 |
| Navier–Stokes | residual | 1.00 | 5 | 1 | 1 | 3 | 0 | 4.64 × 10⁻⁶ | 6.15 × 10⁻⁴ | 0.98 |
| Navier–Stokes | residual | 1.00 | 8 | 1 | 3 | 1 | 0 | 3.80 × 10⁻⁶ | 4.56 × 10⁻⁴ | 0.94 |
| Navier–Stokes | residual | 1.12 | 8 | 1 | 1 | 3 | 0 | 6.02 × 10⁻⁶ | 0.026 | 0.96 |
| Navier–Stokes | residual | 1.25 | 8 | 1 | 2 | 2 | 0 | 7.74 × 10⁻⁶ | 5.79 × 10⁻⁴ | 0.99 |
| Navier–Stokes | residual | 1.50 | 8 | 1 | 0 | 4 | 0 | 1.36 × 10⁻⁵ | 4.77 × 10⁻⁵ | 0.98 |
| Navier–Stokes | residual | 2.00 | 8 | 0 | 1 | 2 | 2 | 2.25 × 10⁻⁵ | 3.03 × 10⁻⁴ | 0.92 |
| Navier–Stokes | residual + noise | 0.62 | 8 | 1 | 2 | 1 | 1 | 1.04 × 10⁻⁵ | 8.09 × 10⁻⁴ | 0.9 |
| Navier–Stokes | direct | 0.62 | 8 | 0 | 0 | 0 | 5 | 2.89 × 10⁻⁵ | 3.05 × 10⁻⁴ | 2.49 × 10⁻³ |
| Navier–Stokes | direct | 1.00 | 8 | 0 | 0 | 0 | 5 | 9.45 × 10⁻⁵ | 6.25 × 10⁻⁴ | 1.67 × 10⁻³ |
| cardiac | residual | 0.25 | 8 | 2 | 3 | 0 | 0 | 1.61 × 10⁻⁵ | — | 0.98 |
| cardiac | residual | 0.50 | 8 | 2 | 3 | 0 | 0 | 4.72 × 10⁻⁵ | — | 1 |
| cardiac | residual | 0.63 | 8 | 0 | 5 | 0 | 0 | 6.09 × 10⁻⁵ | — | 1 |
| cardiac | residual | 0.75 | 8 | 0 | 5 | 0 | 0 | 7.69 × 10⁻⁵ | — | 0.99 |
| cardiac | residual | 1.01 | 8 | 0 | 4 | 1 | 0 | 1.07 × 10⁻⁴ | 6.19 × 10⁻³ | 0.99 |
| cardiac | residual | 1.51 | 8 | 0 | 2 | 2 | 1 | 1.82 × 10⁻⁴ | 5.70 × 10⁻³ | 0.99 |
| cardiac | direct | 0.63 | 8 | 0 | 0 | 0 | 5 | 1.89 × 10⁻⁴ | 4.02 × 10⁻⁵ | 0.064 |
| cardiac | direct | 1.01 | 8 | 0 | 2 | 0 | 3 | 4.79 × 10⁻⁴ | 1.22 × 10⁻⁴ | 0.02 |

**Extended Data Table 7 | Repairing an interpolated parametric operator one
spectral property at a time.** Parametric reduced-order models across 13
Reynolds numbers; columns give median |δ| at training and interpolated Reynolds
numbers and the count of stable rollouts. The last two rows agree to within
0.03%: imposing both properties makes δ equal to the precision of the frequency
estimate and nothing else.

| construction | train | interp | stable |
|---|---|---|---|
| entry interpolation | diverges | diverges | 0/13 |
| mode-matched eigen-interpolation | 3.95 × 10⁻³ | 2.53 × 10⁻³ | 4/11 |
| generic stabilisation (clip non-dominant \|λ\| ≤ 1) | 4.07 × 10⁻¹ | 3.91 × 10⁻¹ | 6/11 |
| + neutrality on the dominant pair | 2.83 × 10⁻² | 7.64 × 10⁻² | 11/11 |
| **+ imposed measured frequency** | **4.50 × 10⁻⁴** | **3.65 × 10⁻⁴** | 11/11 |
| *frequency interpolant's own error* | *4.50 × 10⁻⁴* | *3.66 × 10⁻⁴* | — |

**Extended Data Table 8 | Conditioning a wake operator on Reynolds number.**
Median |δ| at training, interpolated and extrapolated Reynolds numbers, over
four seeds. The POD basis is built from training Reynolds numbers only. Median
training loss varies by 1.5× across the three models (5.8 × 10⁻⁶ to 8.5 × 10⁻⁶)
while δ spans three orders of magnitude; seed spread at fixed Reynolds number
runs 1.5–16× for the free operator. The extrapolated Reynolds numbers lie above
Re ≈ 190, where the physical wake is three-dimensional, and reference-period
jitter there grows to 3 × 10⁻⁴ at Re = 220, below which δ is not resolvable.

| model | params | train | interp | extrap |
|---|---|---|---|---|
| free | 37392 | 7.33 × 10⁻⁴ | 7.14 × 10⁻⁴ | 3.38 × 10⁻² |
| free | 176272 | 3.88 × 10⁻⁴ | 5.78 × 10⁻⁴ | 2.68 × 10⁻² |
| anchored (poly ω) | 108080 | 2.55 × 10⁻⁵ | 3.17 × 10⁻⁴ | 1.79 × 10⁻¹ |

**Extended Data Table 9 | The boundary is one of well-posedness.** Two
Stuart–Landau oscillators with diffusive coupling ε, swept across the locking
transition. Half-vs-half instability is the relative disagreement between period
fits on the two halves of one record, quoted here at a record length of 400 time
units. The quasi-periodic entries are window-sensitive — at ε = 0 the same
diagnostic reads 5.4 × 10⁻⁶, 3.8 × 10⁻¹, 1.1 × 10⁻⁵ and 1.3 × 10⁻⁵ at record
lengths of 200, 400, 800 and 1600 — which is why the classification uses a fixed
threshold rather than the size of the disagreement. Below the locking transition
the single-period fit disagrees with itself and the δ it reports measures
nothing, while both true frequencies are recovered to 10⁻³; once locked, the fit
is stable to 10⁻⁹ and its δ agrees with a direct measurement. Replacing the
single scalar with two, err = √(Σₖ wₖ · 4 sin²(π δₖ Nₖ)), gives
observed/predicted 1.04 at ε = 0 but 3.19, 0.81 and 0.68 at ε = 0.1, 0.2 and
0.3, and is worse than the single-frequency law once locked; the two frequency
families carry 0.44–0.56 of the energy at every coupling, so weight imbalance
does not explain it.

| ε | regime | half-vs-half instability | δ from the fit | δ measured properly |
|---|---|---|---|---|
| 0.00 | quasi-periodic | **3.8 × 10⁻¹** | 6.08 × 10⁻¹ | 1.9 × 10⁻³, 1.8 × 10⁻³ |
| 0.10 | quasi-periodic | 1.3 × 10⁻⁴ | 6.11 × 10⁻¹ | 5.4 × 10⁻³, 4.6 × 10⁻³ |
| 0.30 | quasi-periodic | 1.5 × 10⁻⁵ | 1.16 × 10⁻⁴ | 1.0 × 10⁻⁴, 1.2 × 10⁻⁴ |
| 0.50 | locked | **1.8 × 10⁻⁹** | 2.84 × 10⁻⁴ | the same number |
| 0.80 | locked | **9.6 × 10⁻⁹** | 8.80 × 10⁻⁴ | the same number |
