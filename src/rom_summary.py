import json
import numpy as np
d=json.load(open("results/rom/rom_wake.json")); rows=d["rows"]
print("=== DMD single-Re: dominant eigenvalue magnitude ===")
for r in rows:
    if r["kind"]=="single-Re" and r.get("dmd_mag"):
        print(f"  Re={r['Re']:>4} [{r['split']:>6}]  |lambda| = {r['dmd_mag']:.8f}"
              f"   delta = {r['dmd_delta']:+.3e}")
print("\n  (theory: the phase direction is neutrally stable, so the dominant")
print("   discrete-time eigenvalue must sit on the unit circle)")

def med(kind,key,tag):
    g=[r for r in rows if r["kind"]==kind and r["split"]==tag]
    v=[abs(r[key]) for r in g if r[key] is not None]
    return (np.median(v), len(v), len(g)) if v else (None,0,len(g))

print("\n=== FULL COMPARISON: median |delta| and usable horizon ===")
neural = {
 ("single-Re","free"):        (2.59e-5, "5/5"),
 ("single-Re","phase-anch"):  (2.75e-8, "5/5"),
 ("interp","free"):           (5.36e-4, "4/4"),
 ("interp","phase-anch"):     (2.92e-4, "4/4"),
 ("extrap","free"):           (1.81e-2, "4/4"),
 ("extrap","phase-anch"):     (1.74e-1, "4/4"),
}
print("\n-- single operating point (fit and evaluated at the same Re) --")
h="{:>16} {:>13} {:>13} {:>10}".format("method","median |d|","horizon","stable")
print(h); print("-"*len(h))
v,n,t=med("single-Re","dmd_delta","train")
print("{:>16} {:>13.3e} {:>11.0f} P {:>10}".format("DMD (classical)",v,1/(4*v),f"{n}/{t}"))
for lab,key in [("free neural","free"),("phase-anchored","phase-anch")]:
    dv,st=neural[("single-Re",key)]
    print("{:>16} {:>13.3e} {:>11.0f} P {:>10}".format(lab,dv,1/(4*dv),st))
v,n,t=med("single-Re","opinf_delta","train")
print("{:>16} {:>13.3e} {:>11.0f} P {:>10}".format("OpInf (classical)",v,1/(4*v),f"{n}/{t}"))

print("\n-- parametric, held-out Re --")
print(h); print("-"*len(h))
for lab,key in [("phase-anchored","phase-anch"),("free neural","free")]:
    dv,st=neural[("interp",key)]
    print("{:>16} {:>13.3e} {:>11.0f} P {:>10}".format(lab,dv,1/(4*dv),st))
v,n,t=med("parametric","opinf_delta","interp")
print("{:>16} {:>13.3e} {:>11.0f} P {:>10}".format("OpInf (classical)",v,1/(4*v),f"{n}/{t}"))
v,n,t=med("parametric","dmd_delta","interp")
print("{:>16} {:>13.3e} {:>11.0f} P {:>10}".format("DMD (classical)",v,1/(4*v),f"{n}/{t}"))
