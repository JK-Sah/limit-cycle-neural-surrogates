# Limit-cycle neural surrogates

Why learned surrogates of oscillatory systems lose the oscillation, and what to
constrain instead.

Short answer: the error that matters is phase, not amplitude. It is set by a
single scalar — the relative period error δ — which is invisible in the training
loss and untouched by every stability method currently in use. Constraining the
cycle frequency and amplitude directly cuts δ by about 32× at no cost in
accuracy.

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

## Limitations

Stuart–Landau is a two-dimensional toy with an analytically known cycle. None of
this yet shows the effect dominates in a PDE, that δ is large enough to matter at
engineering horizons, or that anchoring survives a latent phase learned from flow
fields rather than handed over directly. The 200-period horizon is too short:
several configurations never decorrelated within it, so those comparisons are
censored and reported that way. The p = 0.20 correlation is weak evidence of no
relationship, not proof of independence.

Cylinder wake at Re = 100 is the next test, where the Strouhal number is known to
three digits and δ can be measured against literature rather than against another
simulation.

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

## Layout

| path | contents |
|---|---|
| `src/t0_falsify.py` | ground truth, neural ODE, Floquet and drift diagnostics, scaling and regularizer sweeps |
| `src/t0_fix.py` | anchored parametrization and the naive-split ablation |
| `src/t0_anchor_sweep.py` | supervision weight against observation noise |
| `src/analyze.py` | aggregation and the prediction tests |
| `cluster/` | Slurm array scripts |
