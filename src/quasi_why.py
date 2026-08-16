import json, glob
import numpy as np
rows=[]
for f in sorted(glob.glob("results/quasi/*.json")):
    rows += json.load(open(f))
live=[r for r in rows if not r.get("diverged") and "delta" in r]
print("Does the single-period diagnostic fail whenever the attractor is a")
print("torus, or only when two comparably-weighted, well-separated")
print("frequencies are present?\n")
h="{:>6} {:>15} {:>20} {:>10} {:>12} {:>14}".format(
    "eps","regime","frequencies","ratio","weights","single/proper")
print(h); print("-"*len(h))
for e in sorted({r["eps"] for r in live}):
    g=[r for r in live if r["eps"]==e]
    w=g[0]["w"]; wt=g[0]["weights"]
    ds=float(np.median([abs(r["delta_single"]) for r in g]))
    if g[0]["locked"]:
        proper=ds; fr=f"{w[0]:.4f}"; ratio="-"
    else:
        d1=float(np.median([abs(r["delta"][0]) for r in g]))
        d2l=[abs(r["delta"][1]) for r in g if len(r["delta"])>1]
        proper=max(d1,float(np.median(d2l)) if d2l else 0)
        fr=f"{w[0]:.4f},{w[1]:.4f}"; ratio=f"{w[1]/w[0]:.4f}"
    wts=",".join(f"{x:.2f}" for x in wt)
    print("{:>6.2f} {:>15} {:>20} {:>10} {:>12} {:>13.1f}x".format(
        e,"LOCKED" if g[0]["locked"] else "quasi",fr,ratio,wts,
        ds/proper if proper>0 else float('nan')))
print("\nweight imbalance = max(w)/min(w); separation = |w2-w1|/w1")
print("{:>6} {:>12} {:>12} {:>14}".format("eps","imbalance","separation","single/proper"))
print("-"*48)
for e in sorted({r["eps"] for r in live}):
    g=[r for r in live if r["eps"]==e]
    if g[0]["locked"]: continue
    wt=g[0]["weights"]; w=g[0]["w"]
    imb=max(wt)/max(min(wt),1e-9); sep=abs(w[1]-w[0])/w[0]
    ds=float(np.median([abs(r["delta_single"]) for r in g]))
    d1=float(np.median([abs(r["delta"][0]) for r in g]))
    d2l=[abs(r["delta"][1]) for r in g if len(r["delta"])>1]
    proper=max(d1,float(np.median(d2l)) if d2l else 0)
    print("{:>6.2f} {:>12.1f} {:>12.3f} {:>13.1f}x".format(
        e,imb,sep,ds/proper if proper>0 else float('nan')))
