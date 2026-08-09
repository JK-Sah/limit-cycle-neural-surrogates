import json, glob
import numpy as np
rows=[]
for f in sorted(glob.glob("results/arch/*.json")):
    rows += json.load(open(f))["rows"]
meta=json.load(open(sorted(glob.glob("results/arch/*.json"))[0]))
print(f"=== ARCHITECTURE BREADTH, Re={meta['Re']}, {meta['modes']} POD modes ===")
print(f"    reference period {meta['T_lattice']:.3f} lattice steps\n")
order=["DMD (least squares)","linear-sgd","ode-mlp","ode-deep",
       "ar-residual","ar-direct","gru"]
h="{:>20} {:>9} {:>10} {:>11} {:>11} {:>10} {:>8}".format(
    "model","params","med mse","med |delta|","spread","horizon","stable")
print(h); print("-"*len(h))
for name in order:
    g=[r for r in rows if r["arch"]==name]
    if not g: continue
    ds=[abs(r["delta"]) for r in g if r["delta"] is not None]
    ms=[r["train_mse"] for r in g if r["train_mse"] is not None]
    if not ds:
        print("{:>20} {:>9} {:>10} {:>11} {:>11} {:>10} {:>8}".format(
            name,g[0]["nparam"],"-","all diverged","-","-",f"0/{len(g)}"))
        continue
    md=np.median(ds)
    sp=f"{max(ds)/min(ds):.1f}x" if len(ds)>1 and min(ds)>0 else "-"
    print("{:>20} {:>9} {:>10} {:>11.3e} {:>11} {:>8.0f} P {:>4}/{:<3}".format(
        name,g[0]["nparam"],f"{np.median(ms):.2e}" if ms else "-",
        md,sp,1/(4*md),len(ds),len(g)))

print("\n--- the control: same hypothesis class, different fit ---")
for name in ["DMD (least squares)","linear-sgd"]:
    g=[abs(r["delta"]) for r in rows if r["arch"]==name and r["delta"] is not None]
    if g: print(f"  {name:22s} median |delta| = {np.median(g):.3e}"
                f"   horizon {1/(4*np.median(g)):>8.0f} P")
d=[abs(r["delta"]) for r in rows if r["arch"]=="DMD (least squares)" and r["delta"] is not None]
l=[abs(r["delta"]) for r in rows if r["arch"]=="linear-sgd" and r["delta"] is not None]
if d and l:
    print(f"  ratio (sgd / least squares) = {np.median(l)/np.median(d):.1f}x")

print("\n--- does training loss track delta across architectures? ---")
pts=[(r["train_mse"],abs(r["delta"])) for r in rows
     if r["train_mse"] is not None and r["delta"] is not None]
if len(pts)>3:
    from scipy.stats import spearmanr
    m,dd=zip(*pts)
    rho,p=spearmanr(np.log(m),np.log(dd))
    print(f"  Spearman rho(log mse, log |delta|) = {rho:+.3f}  p = {p:.3f}"
          f"   (n={len(pts)})")
