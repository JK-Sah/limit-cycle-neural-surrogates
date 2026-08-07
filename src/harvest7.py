import json, glob
import numpy as np
rows=[]
for f in sorted(glob.glob("results/parwake2/*.json")):
    rows += json.load(open(f))["rows"]
meta=json.load(open(sorted(glob.glob("results/parwake2/*.json"))[0]))
TR,IN,EX = meta["train_re"], meta["interp_re"], meta["extrap_re"]
nseed=len({r["seed"] for r in rows})
print(f"=== PARAMETRIC WAKE, frozen interpolants ({nseed} seeds, 6000 epochs) ===")
h="{:>22} {:>5} {:>8} {:>10} {:>11} {:>11} {:>11}".format(
    "model","width","params","med mse","train","interp","extrap")
print(h); print("-"*len(h))
order=sorted({(r["kind"],r["width"]) for r in rows}, key=lambda t:(t[0]!="free",t[0],t[1]))
for kind,w in order:
    g=[r for r in rows if r["kind"]==kind and r["width"]==w]
    c=[]
    for tag in ["train","interp","extrap"]:
        ds=[abs(r["delta"]) for r in g if r["split"]==tag and r["delta"] is not None]
        c.append("{:.2e}".format(np.median(ds)) if ds else "n/a")
    print("{:>22} {:>5} {:>8} {:>10.2e} {:>11} {:>11} {:>11}".format(
        kind,w,g[0]["nparam"],np.median([r["train_mse"] for r in g]),*c))

print("\n--- usable horizons (periods to pi/2 phase error) ---")
for kind,w in order:
    g=[r for r in rows if r["kind"]==kind and r["width"]==w]
    c=[]
    for tag in ["train","interp","extrap"]:
        ds=[abs(r["delta"]) for r in g if r["split"]==tag and r["delta"] is not None]
        c.append("{:>8.0f} P".format(1/(4*np.median(ds))) if ds else "     n/a")
    print("  {:>22} w={:<4d} train={} interp={} extrap={}".format(kind,w,*c))

print("\n--- does the model hit the delta predicted in closed form? ---")
h="{:>22} {:>8} {:>13} {:>13} {:>9}".format(
    "model","split","predicted","measured","ratio")
print(h); print("-"*len(h))
for kind,w in [k for k in order if k[0]!="free"]:
    for tag in ["train","interp","extrap"]:
        g=[r for r in rows if r["kind"]==kind and r["width"]==w
           and r["split"]==tag and r["delta"] is not None
           and r.get("delta_predicted") is not None]
        if not g: continue
        p=np.median([abs(r["delta_predicted"]) for r in g])
        m=np.median([abs(r["delta"]) for r in g])
        print("{:>22} {:>8} {:>13.3e} {:>13.3e} {:>9.2f}".format(kind,tag,p,m,m/p))

print("\n--- seed spread (max/min |delta| at fixed config+split) ---")
for kind,w in order:
    out=[]
    for tag in ["train","interp","extrap"]:
        per=[]
        for R in (TR if tag=="train" else IN if tag=="interp" else EX):
            ds=[abs(r["delta"]) for r in rows if r["kind"]==kind and r["width"]==w
                and r["Re"]==R and r["delta"] is not None]
            if len(ds)>1 and min(ds)>0: per.append(max(ds)/min(ds))
        out.append("{:>7.1f}x".format(np.median(per)) if per else "    n/a")
    print("  {:>22} w={:<4d} train={} interp={} extrap={}".format(kind,w,*out))
