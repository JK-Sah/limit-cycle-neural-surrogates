# Limit-cycle neural surrogates

Why learned surrogates of oscillatory systems lose the oscillation, and what to
constrain instead.

Short answer: the error that matters is phase, not amplitude. It is set by a
single scalar — the relative period error δ — which is invisible in the training
loss and untouched by every stability method currently in use. Carrying phase as
an explicit coordinate with a measured frequency makes the rollout period exact
by construction.

Two results cut against the obvious story and are reported anyway. On a cylinder
wake at Re = 100 a well-trained free model already reaches δ ≈ 2 × 10⁻⁵, so phase
drift does not limit it. And the first anchored parametrization tried here made
δ *worse*, for a reason worth knowing.

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

St converges to ≈0.169 rather than Williamson's unconfined 0.1643. The obvious
explanation, blockage, is **wrong**: opening the domain from 5% to 1.7% blockage
at fixed D leaves St at ~0.169 and does not move it monotonically.

| ny/D | blockage | St | vs 0.1643 |
|---|---|---|---|
| 20 | 5.00% | 0.16851 | +2.56% |
| 30 | 3.33% | 0.16904 | +2.89% |
| 40 | 2.50% | 0.16930 | +3.05% |
| 60 | 1.67% | 0.16878 | +2.72% |

The remaining suspect is compressibility: these runs use Ma = 0.173, above the
Ma < 0.1 that lattice Boltzmann practice calls for. A Mach study at fixed
relaxation time (D scaled inversely with inflow speed) is running. Until it
resolves, the +2.9% offset is an open discrepancy and is reported as one.

None of this affects the δ results, which measure a surrogate against the
solver's own period. Period jitter near 5 × 10⁻⁷ is what those need: the
reference period is good to six digits.

**Drag is retracted.** The momentum-exchange estimate gives Cd = 1.05, 0.43,
0.04, −0.20 over D = 16…40, and negative drag is impossible. True drag is ~0.27
in lattice units against a raw link sum of ~15, so the estimate rides on 2%
cancellation, and any systematic error in the link set grows with the perimeter
and overwhelms it. Flagged in the code and excluded from output. Strouhal comes
from the probe and is unaffected.

## Limitations

Stuart–Landau is a two-dimensional toy with an analytically known cycle. None of
this yet shows the effect dominates in a PDE, that δ is large enough to matter at
engineering horizons, or that anchoring survives a latent phase learned from flow
fields rather than handed over directly. The 200-period horizon is too short:
several configurations never decorrelated within it, so those comparisons are
censored and reported that way. The p = 0.20 correlation is weak evidence of no
relationship, not proof of independence.

For the wake, the honest summary is that the diagnostic transfers and the warning
does not. The phase law, the loss decoupling and the insensitivity to capacity
all reproduce on Navier–Stokes, but δ for a well-trained free model is ~2 × 10⁻⁵,
so phase drift is not what limits a cylinder-wake surrogate. A Re = 100 wake is
genuinely low-dimensional and near-harmonic — closer to Stuart–Landau than to a
hard problem — and 8 POD modes on a single noise-free trajectory is the easiest
version of the task.

Where δ should become hard to control is where the frequency itself is hard to
pin: operators generalising across Reynolds number, higher Re with broadband wake
dynamics, and flexible FSI where the structure sets its own timescale. Those are
the next cases. Until one of them shows a large δ, phase-anchoring is worth using
for the guarantee and the training-cost saving rather than as a rescue.

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
