**To the Editor, *Nature Machine Intelligence***

**Re:** *Two spectral properties govern long-horizon fidelity in surrogates of oscillatory flows, and standard practice preserves neither*

Dear Editor,

I am submitting the manuscript above for consideration as an Article.

Learned surrogates of oscillatory systems lose accuracy over long rollouts, and the field's response has been a family of stability mechanisms: energy penalties, Lipschitz constraints, noise injection during training. The manuscript shows these act on the wrong direction. A stable limit cycle has one Floquet multiplier exactly equal to unity along the orbit, forced by time-translation invariance, with the rest strictly inside the unit circle. Amplitude error therefore self-corrects while phase error has no restoring force at all, so long-horizon behaviour is set by a single scalar — the relative period error δ — which is measurable from a ten-period rollout and predicts decorrelation thousands of periods later.

Three results seem to me worth an editor's attention.

The training loss carries no information about δ at the level a practitioner selects on. Across 75 models on a Stuart–Landau oscillator and twelve surrogate configurations on a cylinder wake, δ spans six orders of magnitude; loss orders the model classes weakly (ρ = +0.62, p = 0.033, taking each configuration as one observation) and orders individual models not at all, since at fixed architecture, data and objective δ varies by up to 6 × 10⁴× across random seeds alone. Sixty-four times more data does not move it.

This is not a pathology of neural networks. Plain least-squares dynamic mode decomposition recovers the same neutral multiplier to eight decimals, and the effect appears in classical reduced-order models, autoregressive networks, recurrent networks and Fourier neural operators alike. The FNO results matter for your readership specifically: the configuration reaching the *lowest* training loss is the only one that cannot complete a rollout, and adding noise injection restores rollout stability while leaving the period error untouched at a ratio of 1.09 — a stability method fixing stability and doing nothing for phase, which is the paper's argument in one experiment.

The diagnosis comes with a constructive result. Two properties must be imposed and neither alone suffices; impose both and δ collapses to the precision of the frequency estimate, reproducing it to within 0.03% on a classical reduced-order model. Because the period is then fixed by construction, the usable horizon is computable in closed form before any model is fitted.

The manuscript reports what failed alongside what worked: an anchoring scheme of my own that is worse than doing nothing outside the training range, a two-frequency generalisation of the error law that does not hold, and the quasi-periodic boundary where the single-scalar description becomes ill-posed and a δ quoted there measures nothing. It also supplies the test that detects that case.

The work is my own and has not been published elsewhere or submitted for consideration to another journal. I declare no competing interests. All code, Slurm scripts and result files are in a repository I will make public on acceptance, and I can provide reviewers with access at any point during review.

Yours sincerely,

J. K. Sah
Department of Mechanical and Industrial Engineering
Rochester Institute of Technology
js8472@rit.edu
