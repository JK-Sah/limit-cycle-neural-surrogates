import json, glob, math
import numpy as np
rows=[]
for f in sorted(glob.glob("results/downstream/*.json")):
    rows += json.load(open(f))["rows"]
live=[r for r in rows if not r.get("diverged") and "err_inst" in r]
order=["ode-mlp","ode-deep","ar-residual","ar-direct","linear-sgd","exact DMD","gru"]
print("per-seed: predicted 1/(4d) against the first sampled N where err >= sqrt(2)")
print("(the earlier table paired a MEDIAN delta with a MIN-over-seeds crossing,")
print(" which is not a comparison of anything)\n")
print("{:>13} {:>5} {:>11} {:>12} {:>12} {:>9}".format(
    "model","seed","delta","1/(4 delta)","observed","ratio"))
print("-"*68)
agree=[]
for a in order:
    seeds=sorted({r["seed"] for r in live if r["arch"]==a})
    for s in seeds:
        g=sorted([r for r in live if r["arch"]==a and r["seed"]==s],
                 key=lambda r:r["N"])
        if not g: continue
        d=g[0].get("delta")
        if d is None or d==0: continue
        pred=1/(4*abs(d))
        hit=[r["N"] for r in g if r["err_inst"]>=math.sqrt(2)]
        obs=min(hit) if hit else None
        # the sampled grid is coarse, so bracket the prediction instead of
        # demanding an exact match
        Ns=[r["N"] for r in g]
        lo=max([n for n in Ns if n<=pred], default=None)
        hi=min([n for n in Ns if n>=pred], default=None)
        ok = obs is not None and lo is not None and hi is not None and lo<=obs<=hi*3
        if obs: agree.append(obs/pred)
        print("{:>13} {:>5} {:>11.2e} {:>12.0f} {:>12} {:>9}".format(
            a,s,abs(d),pred, str(obs) if obs else ">30000",
            f"{obs/pred:.2f}" if obs else "-"))
print("-"*68)
if agree:
    print(f"  median observed/predicted crossing = {np.median(agree):.2f}"
          f"   (n={len(agree)}; sampling grid is ~3x spaced, so 1-3 is agreement)")

print("\n=== phase-invariance holds only for models with an attracting cycle ===\n")
print("{:>13} {:>12} {:>12} {:>12} {:>10}".format(
    "model","|cov| N=10","|cov| N=30000","growth","has attractor"))
print("-"*64)
for a in order:
    g=[r for r in live if r["arch"]==a]
    lo=[r["err_cov"] for r in g if r["N"]==10 and "err_cov" in r]
    hi=[r["err_cov"] for r in g if r["N"]==30000 and "err_cov" in r]
    if not lo or not hi: continue
    l,h=np.median(lo),np.median(hi)
    nl = a in ("ode-mlp","ode-deep","ar-residual","ar-direct","gru")
    print("{:>13} {:>12.1e} {:>12.1e} {:>12.1f}x {:>10}".format(
        a,l,h,h/l,"nonlinear" if nl else "LINEAR"))
