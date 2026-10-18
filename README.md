# Limit-cycle neural surrogates

How long does a learned surrogate of an oscillating system stay in step with
it, and what decides that?

**Short answer.** Two parts of the surrogate's own Floquet spectrum, and
standard practice measures neither.

1. A stable limit cycle has one multiplier at exactly 1 (the phase direction,
   forced by time-translation invariance) and the rest inside the unit circle.
   So amplitude error self-corrects and phase error never does, and a
   surrogate *with* a stable cycle loses fidelity at a rate set by one scalar,
   its relative period error δ. The usable horizon is 1/(4δ) periods.
2. The transverse multipliers decide whether there is a stable cycle at all.
   Residual (increment-predicting) Fourier neural operators mostly do not have
   one: their transverse multipliers sit at 0.97 per period. Direct
   (next-state) operators do: 0.02 per period.

The manuscript is [`nmi_article.md`](nmi_article.md). Every number in it is
reproducible from the per-run records in `results/`.

## Results

| claim | evidence | where |
|---|---|---|
| Phase drift is set by the period alone | observed/predicted drift rate median 1.0000 over 75 neural ODEs (a consistency check: both come from the learned cycle) | Fig. 1 |
| δ forecasts long-horizon error | δ from 100 periods predicts phase-resolved error to 30,000 periods (per-model ratios 1.02–1.08); time-averaged statistics flat to 1.1× | Fig. 2, ED Tables 1–2 |
| Stability methods act on the wrong direction | energy, Lipschitz and noise penalties never improve δ (p ≥ 0.14); Lipschitz makes it 27.4× worse (p = 0.024) while doubling the transverse multiplier | ED Table 3 |
| Not neural-specific | least-squares DMD recovers the neutral multiplier to within 5 × 10⁻⁸ | ED Table 4 |
| Loss does not order surrogates by δ | configuration-level ρ = +0.02 (p = 0.95) over 14 configurations; within-configuration seed spread 2.1× to 1.4 × 10⁶× | Fig. 3, ED Table 5 |
| Residual field operators lose the cycle | of 120 residual FNOs on a cylinder wake, 2D Navier–Stokes and a cardiac spiral wave, 26 diverge, 41 collapse or lock onto a wrong rhythm, 46 keep an orbit that fails the split-half test, 7 keep a clean cycle; transverse multiplier median 0.97 per period (0 of 95 below 0.5) | Fig. 4a,b, ED Table 6 |
| Direct operators keep it | 28 of 30 keep a clean cycle at the true period; transverse multiplier median 0.019; δ smaller in all six matched comparisons (5× to >1,000×), though they fit 1.4–25× worse | Fig. 4a–c |
| Segment length helps the cycle, not δ | residual wake operators keep a cycle in 3/10 runs below one period vs 18/20 at or above (p = 0.002); cardiac 0/20 vs 4/10 (p = 0.008); no effect on Navier–Stokes; no consistent effect on δ | Fig. 4d |
| Neutrality and frequency must both be imposed | on a parametric ROM, imposing both makes δ equal to the frequency estimate's own error to 0.03% | Fig. 5, ED Table 7 |
| Conditioning on a parameter costs the horizon | ~10⁴ periods at one Reynolds number, 433 interpolating, 9 extrapolating, at unchanged training loss | Fig. 6, ED Table 8 |
| Where the diagnostic stops | on quasi-periodic attractors the single-period fit is ill-posed; split-half refits separate locked (≤ 10⁻⁸) from quasi-periodic (≥ 1.1 × 10⁻⁶) records without overlap | ED Table 9 |

### What did not work, reported anyway

- Anchoring a latent model's phase to a measured frequency: wins in
  distribution, loses outside the training range.
- A period penalty in an operator's training objective: no significant gain on
  the wake (p = 0.18), destroys the Navier–Stokes model at the same weight.
- A two-frequency generalisation of the error law on quasi-periodic attractors.
- An earlier three-seed result that a one-period training objective improves δ
  81-fold on Navier–Stokes. Five seeds under a settled protocol did not
  reproduce it (median |δ| 4.4 × 10⁻⁴ at 25 steps against 3.1 × 10⁻³ at 40).
- A convolutional U-Net: half its seeds never left a loss plateau, so no
  architecture comparison is drawn from it.

### Measurement protocol for field operators

δ of a learned operator is only meaningful once its own long rollout has
settled onto a cycle. Every operator result comes from one protocol: A100 GPUs
in float64, seeded initialisation and segment order (training is
deterministic: repeated configurations reproduce loss and δ to the printed
precision), a 400-period rollout, δ fitted on periods 200–400, and the
split-half test (< 10⁻⁷) applied to the surrogate's own late record. On
100-period rollouts the two quarters of the fitting window disagree by a median
25% of δ for residual operators (0.0% for direct ones), so a 100-period δ of a
residual operator describes a transient.

## Reproducing

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt          # CPU work; GPU runs used torch 2.8.0+cu126
python src/figures.py                    # every figure, from results/
python src/operator_summary.py           # Extended Data Table 6
python src/seglen_analysis.py            # segment-length tests
```

Experiments were run on the RIT SPORC cluster; the Slurm scripts are in
`cluster/`, one seed per array task. The heavy ones:

| display item | scripts | results |
|---|---|---|
| Fig. 1, ED Table 3 | `src/t0_falsify.py`, `cluster/t0_array.sh` | `results/all/` |
| Fig. 2, ED Tables 1–2 | `src/downstream.py`, `src/horizon_perseed.py`, `cluster/downstream_job.sh` | `results/downstream/` |
| Fig. 3, ED Table 5 | `src/arch_breadth.py`, `cluster/arch_job.sh` | `results/arch_v2/` |
| Fig. 4, ED Table 6 | `src/fno.py`, `src/fno_kolm.py`, `src/cardiac.py`, `cluster/seglen_v2.sh`, `cluster/cardiac_data.sh`; multipliers `src/operator_floquet.py`, `cluster/operator_floquet.sh` | `results/seglen_v2/`, `results/floquet_ops/`, `results/cardiac/` |
| Fig. 5, ED Table 7 | `src/rom_parametric.py` | `results/rom/` |
| Fig. 6, ED Table 8 | `src/param_wake.py` | `results/parwake_v2/` |
| ED Fig. 1 | `src/kolmogorov.py` | `results/kolmogorov/`, `results/kolm_onset/`, `results/kolm_data/` |
| ED Table 9 | `src/quasi_split.py`, `cluster/quasi_split.sh` | `results/quasi_split/` |

Raw snapshot fields are regenerable from the solvers and are not in the repo.
The trained weights of the 170 field operators (4.9 GB) are not in the repo
either; they will be archived with the Zenodo deposit and are available on
request. The projected 400-period rollouts of every operator are in
`results/seglen_v2/coefs/`.

## Layout

| path | contents |
|---|---|
| `src/period.py` | Fourier period estimator, validated to 1e-10 |
| `src/t0_falsify.py`, `src/t0_fix.py`, `src/t0_anchor_sweep.py` | Stuart–Landau ground truth, neural ODEs, Floquet and drift diagnostics, regulariser sweeps, phase anchoring |
| `src/cylinder_lbm.py`, `src/cyl_latent.py` | D2Q9 lattice-Boltzmann wake; POD and latent ODEs |
| `src/kolmogorov.py` | pseudo-spectral 2D Navier–Stokes in the FNO-benchmark forcing |
| `src/cardiac.py` | Aliev–Panfilov spiral wave, with period convergence checks |
| `src/fno.py`, `src/fno_kolm.py`, `src/unet.py` | field operators (residual and direct), long-rollout δ with split-half test |
| `src/operator_floquet.py` | Lyapunov exponents / Floquet multipliers of a trained operator |
| `src/operator_summary.py`, `src/seglen_analysis.py` | outcome classification, tests, Extended Data Table 6 |
| `src/arch_breadth.py`, `src/rom_wake.py`, `src/rom_parametric.py`, `src/optdmd*.py` | architecture breadth, DMD and the two-property repair |
| `src/param_wake.py`, `src/param_osc.py` | parametric operators |
| `src/quasiperiodic.py`, `src/quasi_split.py` | coupled oscillators; split-half well-posedness test |
| `src/figures.py` | every figure, regenerated from the result files |
| `src/filament.py`, `src/ibm.py`, `src/iblbm.py`, `src/flapping.py` | flexible-filament FSI solver, built and validated, not used in the paper (its period does not converge) |
| `cluster/` | Slurm array scripts |

MIT licence.
