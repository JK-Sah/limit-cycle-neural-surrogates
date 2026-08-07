import json, glob, math
import numpy as np
def st_w(Re): return -3.3265/Re + 0.1816 + 1.6e-4*Re
TR=[80,100,120,140,160,180]; IN=[90,110,130,150,170]; EX=[200,220]
h="{:>5} {:>7} {:>10} {:>10} {:>10} {:>10} {:>9}".format(
    "Re","split","tau","St_meas","St_Will","rel diff","jitter")
print(h); print("-"*len(h))
diffs=[]
for f in sorted(glob.glob("results/resweep/Re*.json"),
                key=lambda s:int(''.join(c for c in s.split('/')[-1] if c.isdigit()))):
    d=json.load(open(f)); R=int(d["Re"])
    tag="train" if R in TR else "interp" if R in IN else "extrap"
    rd=(d["St"]-st_w(R))/st_w(R)*100
    diffs.append(rd)
    print("{:>5} {:>7} {:>10.4f} {:>10.5f} {:>10.5f} {:>+9.2f}% {:>9.1e}".format(
        R, tag, d["tau"], d["St"], st_w(R), rd, d["T_std"]/d["T_lattice"]))
print("\n  mean offset {:+.2f}%  spread {:.2f}%  (Williamson correlation is "
      "quoted for 47<Re<180)".format(np.mean(diffs), np.std(diffs)))
print("  monotonic in Re: {}".format(
    all(json.load(open(f"results/resweep/Re{a}.json"))["St"]
        < json.load(open(f"results/resweep/Re{b}.json"))["St"]
        for a,b in zip(sorted(TR+IN+EX)[:-1], sorted(TR+IN+EX)[1:]))))
