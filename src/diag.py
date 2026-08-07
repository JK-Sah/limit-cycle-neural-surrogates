import json, glob
import numpy as np
rows=[]
for f in sorted(glob.glob("results/parwake2/*.json")):
    rows += json.load(open(f))["rows"]
print("Do the two interpolants give DIFFERENT extrapolation errors?")
print("If the error came from omega they must differ (predicted 5.99e-3 vs")
print("1.49e-3, a 4x gap). If it comes from the decoder they will coincide.\n")
h="{:>5} {:>6} {:>14} {:>14} {:>9}".format("Re","seed","poly4","williamson","ratio")
print(h); print("-"*len(h))
for R in [200,220]:
    for s in sorted({r["seed"] for r in rows}):
        v={}
        for k in ["anchored[poly4]","anchored[williamson]"]:
            g=[abs(r["delta"]) for r in rows if r["kind"]==k and r["Re"]==R
               and r["seed"]==s and r["delta"] is not None]
            v[k]=g[0] if g else None
        if v["anchored[poly4]"] and v["anchored[williamson]"]:
            a,b=v["anchored[poly4]"],v["anchored[williamson]"]
            print("{:>5} {:>6} {:>14.4e} {:>14.4e} {:>9.2f}".format(R,s,a,b,a/b))
print("\nSame comparison at an INTERPOLATED Re, where the decoder is in-distribution:")
print(h); print("-"*len(h))
for R in [110,150]:
    for s in sorted({r["seed"] for r in rows}):
        v={}
        for k in ["anchored[poly4]","anchored[williamson]"]:
            g=[abs(r["delta"]) for r in rows if r["kind"]==k and r["Re"]==R
               and r["seed"]==s and r["delta"] is not None]
            v[k]=g[0] if g else None
        if v["anchored[poly4]"] and v["anchored[williamson]"]:
            a,b=v["anchored[poly4]"],v["anchored[williamson]"]
            print("{:>5} {:>6} {:>14.4e} {:>14.4e} {:>9.2f}".format(R,s,a,b,a/b))
