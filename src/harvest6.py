import json, glob, math
import numpy as np
rows=[]
for f in sorted(glob.glob("results/parwake/*.json")):
    rows += json.load(open(f))["rows"]
meta=json.load(open(sorted(glob.glob("results/parwake/*.json"))[0]))
TR,IN,EX = meta["train_re"], meta["interp_re"], meta["extrap_re"]
print("=== PARAMETRIC WAKE ({} seeds present) ===".format(
    len({r["seed"] for r in rows})))
h="{:>15} {:>5} {:>8} {:>7} {:>10} {:>11} {:>11} {:>11}".format(
    "model","width","params","seeds","med mse","train","interp","extrap")
print(h); print("-"*len(h))
for kind,w in sorted({(r["kind"],r["width"]) for r in rows},
                     key=lambda t:(t[0]!="free",t[1])):
    g=[r for r in rows if r["kind"]==kind and r["width"]==w]
    c=[]
    for tag in ["train","interp","extrap"]:
        ds=[abs(r["delta"]) for r in g if r["split"]==tag and r["delta"] is not None]
        c.append("{:.2e}".format(np.median(ds)) if ds else "n/a")
    print("{:>15} {:>5} {:>8} {:>7} {:>10.2e} {:>11} {:>11} {:>11}".format(
        kind,w,g[0]["nparam"],len({r["seed"] for r in g}),
        np.median([r["train_mse"] for r in g]),*c))

print("\n--- horizons (periods to pi/2 phase error) ---")
for kind,w in sorted({(r["kind"],r["width"]) for r in rows},
                     key=lambda t:(t[0]!="free",t[1])):
    g=[r for r in rows if r["kind"]==kind and r["width"]==w]
    c=[]
    for tag in ["train","interp","extrap"]:
        ds=[abs(r["delta"]) for r in g if r["split"]==tag and r["delta"] is not None]
        c.append("{:.0f} P".format(1/(4*np.median(ds))) if ds else "n/a")
    print("  {:>15} w={:<4d} train={:>9} interp={:>9} extrap={:>9}".format(kind,w,*c))

print("\n--- per-Re median |delta|, and SEED SPREAD ---")
h="{:>5} {:>7} {:>13} {:>10} {:>13} {:>10}".format(
    "Re","split","free288 med","spread","anchored med","spread")
print(h); print("-"*len(h))
for R in sorted(TR+IN+EX):
    tag = "train" if R in TR else "interp" if R in IN else "extrap"
    out=[]
    for kind,w in [("free",288),("phase-anchored",128)]:
        ds=[abs(r["delta"]) for r in rows if r["kind"]==kind and r["width"]==w
            and r["Re"]==R and r["delta"] is not None]
        out += (["{:.2e}".format(np.median(ds)),
                 "{:.0f}x".format(max(ds)/min(ds)) if len(ds)>1 and min(ds)>0 else "-"]
                if ds else ["n/a","-"])
    print("{:>5} {:>7} {:>13} {:>10} {:>13} {:>10}".format(R,tag,*out))
