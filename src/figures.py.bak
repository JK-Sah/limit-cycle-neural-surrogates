"""
Figures for the paper, regenerated from result files.

Everything here reads the JSON that the sweeps wrote and draws from it. Nothing
is transcribed by hand, so a re-run of any experiment propagates into the
figures by re-running this script -- which matters in a study where several
tables have already been re-measured once.

Colour. Five categorical hues from the Okabe-Ito set, chosen by computing
pairwise separation in OKLab under normal vision and simulated deuteranopia,
protanopia and tritanopia rather than by eye. Five is the ceiling: the best
six-hue subset drops to a worst-case dE of 7.9, below the usable threshold,
while this five reaches 11.2. Yellow is excluded despite being separable
because its contrast on white is 1.6:1 and these are print figures. Where more
than five categories are needed the extra distinction is carried by marker
shape and dash pattern, never by a sixth hue -- which also keeps the figures
readable in greyscale.
"""
import argparse, glob, itertools, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

INK, MUTED = "#1a1a1a", "#6b6b6b"
# Blue and green are NOT both usable: they collapse to dE 5.4 under simulated
# tritanopia. This is the best five-hue subset with yellow excluded, at 11.2.
C = {"black": "#000000", "skyblue": "#56B4E9", "green": "#009E73",
     "vermillion": "#D55E00", "purple": "#CC79A7"}
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]

plt.rcParams.update({
    "figure.dpi": 160, "savefig.dpi": 300, "savefig.bbox": "tight",
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8.5,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#e8e8e8", "grid.linewidth": 0.6,
    "lines.linewidth": 1.4, "legend.frameon": False,
})


def _validate_palette():
    """Recompute CVD separation so the claim in the docstring stays true."""
    def s2l(c): return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)
    def oklab(c):
        M1 = np.array([[.4122214708, .5363325363, .0514459929],
                       [.2119034982, .6806995451, .1073969566],
                       [.0883024619, .2817188376, .6299787005]])
        M2 = np.array([[.2104542553, .7936177850, -.0040720468],
                       [1.9779984951, -2.4285922050, .4505937099],
                       [.0259040371, .7827717662, -.8086757660]])
        return M2 @ np.cbrt(M1 @ c)
    def cvd(c, k):
        T = {"d": np.array([[.625, .375, 0], [.7, .3, 0], [0, .3, .7]]),
             "p": np.array([[.567, .433, 0], [.558, .442, 0], [0, .242, .758]]),
             "t": np.array([[.95, .05, 0], [0, .433, .567], [0, .475, .525]])}[k]
        return np.clip(T @ c, 0, 1)
    lin = [s2l(np.array([int(h.lstrip("#")[i:i+2], 16) / 255 for i in (0, 2, 4)]))
           for h in C.values()]
    w = min(np.linalg.norm(oklab(a if m is None else cvd(a, m))
                           - oklab(b if m is None else cvd(b, m))) * 100
            for a, b in itertools.combinations(lin, 2)
            for m in (None, "d", "p", "t"))
    return w


def _load(pat):
    out = []
    for f in sorted(glob.glob(pat)):
        j = json.load(open(f))
        rows = j["rows"] if isinstance(j, dict) and "rows" in j else j
        out.extend(rows if isinstance(rows, list) else [rows])
    return out


def _save(fig, name, outdir):
    os.makedirs(outdir, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(outdir, f"{name}.{ext}"))
    plt.close(fig)
    print(f"  wrote {outdir}/{name}.pdf/.png")


# ------------------------------------------------------------------ figure 1
def fig1(res, outdir):
    """The phase law: observed drift rate against the rate predicted from delta."""
    rows = _load(f"{res}/all/*.json")
    obs = np.array([r["phase_slope"] for r in rows])
    pred = np.array([r["phase_slope_pred"] for r in rows])
    ratio = np.array([r["phase_slope_ratio"] for r in rows])

    # log-log on magnitudes: delta spans four decades, and on linear axes every
    # model but the few worst collapses onto the origin. Sign is carried by the
    # marker so nothing is hidden by taking absolute values.
    fig, ax = plt.subplots(figsize=(3.4, 3.0))
    lo = min(np.abs(np.r_[obs, pred])[np.abs(np.r_[obs, pred]) > 0].min() * 0.5, 1e-6)
    hi = np.abs(np.r_[obs, pred]).max() * 2
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1.0, ls=(0, (4, 3)),
            zorder=1, label="y = x")
    for sgn, mk, lab in ((1, "o", "δ > 0"), (-1, "^", "δ < 0")):
        m = np.sign(pred) == sgn
        if m.sum():
            ax.scatter(np.abs(pred[m]), np.abs(obs[m]), s=16, marker=mk,
                       facecolor=C["skyblue"], edgecolor="white",
                       linewidth=0.5, alpha=0.85, zorder=3,
                       label=f"{lab}  (n = {m.sum()})")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.set_xlabel(r"|predicted drift rate|  $2\pi|\delta|/T$")
    ax.set_ylabel(r"|observed drift rate|  $|d\Delta\theta/dt|$")
    ax.set_title("Phase drift is fully accounted for by the period error")
    ax.legend(loc="upper left")

    ins = fig.add_axes([0.60, 0.20, 0.30, 0.22])
    ins.hist(ratio, bins=18, color=C["vermillion"], edgecolor="white", lw=0.4)
    ins.axvline(1.0, color=INK, lw=1.0)
    ins.set_xlabel("observed / predicted", fontsize=6)
    ins.set_yticks([])
    ins.tick_params(labelsize=6)
    ins.set_title(f"median {np.median(ratio):.4f}", fontsize=6)
    for sp in ("top", "right", "left"):
        ins.spines[sp].set_visible(False)
    ins.grid(False)
    _save(fig, "fig1_phase_law", outdir)


# ------------------------------------------------------------------ figure 2
def fig2(res, outdir):
    """What the period error costs: the phase-resolved / phase-invariant split."""
    rows = [r for r in _load(f"{res}/downstream/*.json") if not r.get("diverged")]
    archs = ["ode-mlp", "ode-deep", "ar-residual", "linear-sgd", "exact DMD"]
    colors = [C["skyblue"], C["green"], C["purple"], C["vermillion"], C["black"]]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))
    for ax, key, ttl, ylab in (
            (axes[0], "err_inst", "Phase-resolved error follows $2|\\sin(\\pi\\delta N)|$",
             "relative RMS error at instant"),
            (axes[1], "err_cov", "Phase-invariant statistics stay flat -- with an attractor",
             "covariance error")):
        for a, col, mk in zip(archs, colors, MARKERS):
            sub = sorted([r for r in rows if r["arch"] == a], key=lambda r: r["N"])
            if not sub:
                continue
            N = np.array([r["N"] for r in sub], float)
            v = np.array([r[key] for r in sub], float)
            Nu = np.unique(N)
            med = np.array([np.median(v[N == n]) for n in Nu])
            ax.plot(Nu, med, marker=mk, ms=3.5, color=col, label=a,
                    markeredgecolor="white", markeredgewidth=0.4)
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlabel("rollout length  N  (shedding periods)")
        ax.set_ylabel(ylab); ax.set_title(ttl, fontsize=7.5)
    axes[0].axhline(np.sqrt(2), color=MUTED, lw=1.0, ls=(0, (4, 3)))
    axes[0].annotate(r"$\sqrt{2}$ = decorrelated", (1.2, np.sqrt(2) * 1.15),
                     color=MUTED, fontsize=6)
    axes[1].legend(loc="upper left", ncol=1)
    _save(fig, "fig2_what_it_costs", outdir)


# ------------------------------------------------------------------ figure 3
def fig3(res, outdir):
    """Loss is blind: training error against delta, within and across families.

    The figure has to carry a two-level claim, because a single scatter of every
    run supports the wrong reading. Pooled over runs the correlation looks
    strong and significant, but seeds within one configuration are not
    independent observations, so that number is inflated. Taking each
    configuration as one point gives the defensible statistic. Both are drawn:
    faint markers for individual runs, solid markers for configuration medians,
    and a vertical bar spanning each configuration's seed range. The bars are
    the argument -- they are taller than the trend the medians follow.
    """
    import statistics as st
    from scipy.stats import spearmanr

    fam = {"ode-mlp": ("continuous", C["skyblue"]), "ode-deep": ("continuous", C["skyblue"]),
           "ar-direct": ("discrete", C["green"]), "ar-residual": ("discrete", C["green"]),
           "gru": ("recurrent", C["purple"]), "linear-sgd": ("linear", C["vermillion"]),
           "fno": ("operator", C["black"]), "deeponet": ("operator", C["black"]),
           "fnoNS": ("operator, PDE field", C["vermillion"])}

    rows = [r for r in _load(f"{res}/arch_v2/*.json")
            if r.get("train_mse") and r.get("delta")]
    for pat, tag in ((f"{res}/fno/seg*.json", "fno"),
                     (f"{res}/fno_seeds/*.json", "fno"),
                     (f"{res}/fno_kolm/*.json", "fnoNS"),
                     (f"{res}/deeponet_v2/*.json", "deeponet")):
        for f in sorted(glob.glob(pat)):
            base = os.path.basename(f).split("_s")[0].replace(".json", "")
            j = json.load(open(f))
            for r in j["rows"]:
                if r.get("train_mse") and r.get("delta"):
                    r = dict(r); r["arch"] = f"{tag}-{base}"; rows.append(r)

    grp = {}
    for r in rows:
        grp.setdefault(r["arch"], []).append((r["train_mse"], abs(r["delta"])))

    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    pts = []
    for i, k in enumerate(sorted(grp)):
        base = k.split("-")[0] if k.startswith(("fno", "deeponet")) else k
        label, col = fam.get(base, ("other", MUTED))
        v = grp[k]
        x = [a for a, _ in v]; y = [b for _, b in v]
        mk = MARKERS[i % len(MARKERS)]
        ax.scatter(x, y, s=9, color=col, marker=mk, alpha=0.30,
                   edgecolor="none", zorder=2)
        mx, my = st.median(x), st.median(y)
        ax.plot([mx, mx], [min(y), max(y)], color=col, lw=1.0, alpha=0.55, zorder=3)
        ax.scatter([mx], [my], s=42, color=col, marker=mk, edgecolor="white",
                   linewidth=0.7, zorder=4)
        pts.append((mx, my))

    # Legend by family, not by configuration. Colour already carries family and
    # twelve entries would cover the data; the markers separate configurations
    # within a family for anyone reading closely.
    from matplotlib.lines import Line2D
    fams = {}
    for k in grp:
        base = k.split("-")[0] if k.startswith(("fno", "deeponet")) else k
        lab, col = fam.get(base, ("other", MUTED))
        fams.setdefault(lab, col)
    handles = [Line2D([], [], marker="o", ls="none", color=c, label=l,
                      markersize=5, markeredgecolor="white")
               for l, c in sorted(fams.items())]

    rho_c, p_c = spearmanr(np.log10([a for a, _ in pts]),
                           np.log10([b for _, b in pts]))
    spreads = [max(b for _, b in v) / min(b for _, b in v)
               for v in grp.values() if min(b for _, b in v) > 0]

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("training MSE")
    ax.set_ylabel(r"$|\delta|$")
    ax.set_title("Loss orders model classes weakly, and models not at all",
                 fontsize=8)
    ax.annotate(
        f"across configurations: $\\rho$ = {rho_c:+.2f} (p = {p_c:.3f}, n = {len(pts)})\n"
        f"within one configuration: {min(spreads):.1f}× to "
        f"{max(spreads)/10**int(np.log10(max(spreads))):.0f} × 10"
        f"{chr(0x2070+int(np.log10(max(spreads))))}× in $\\delta$",
        (0.03, 0.03), xycoords="axes fraction", fontsize=6.2, color=MUTED,
        linespacing=1.5)
    ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=6,
              handletextpad=0.3, columnspacing=1.0)
    _save(fig, "fig3_loss_is_blind", outdir)


# ------------------------------------------------------------------ figure 4
def fig4(res, outdir):
    """Two properties: the eigenvalue plane, repaired one property at a time."""
    src = f"{res}/rom/rom_parametric_v3.json"
    if not os.path.exists(src):
        print("  skipping fig4: rom_parametric_v3.json not present yet")
        return
    rows = [r for r in json.load(open(src)) if r.get("spectrum")]
    stages = [("entry", "entry interpolation"), ("stab", "generic stabilisation"),
              ("neutral", "+ neutrality"), ("neutral+freq", "+ measured frequency")]
    cols = [C["vermillion"], C["purple"], C["skyblue"], C["green"]]

    fig, axes = plt.subplots(1, 4, figsize=(7.2, 2.1))
    for ax, (key, ttl), col in zip(axes, stages, cols):
        ax.add_patch(Circle((0, 0), 1.0, fill=False, color=MUTED, lw=1.0,
                            ls=(0, (4, 3))))
        for r in rows:
            sp = np.array(r["spectrum"][key])
            ax.scatter(sp[:, 0], sp[:, 1], s=7, color=col, alpha=0.7,
                       edgecolor="none")
        # One axis range for all four panels. Rescaling each to its own data
        # would hide the shrinkage the figure exists to show; points outside
        # the frame are counted in the title instead of being silently dropped.
        rad = max(np.linalg.norm(np.array(r["spectrum"][key]), axis=1).max()
                  for r in rows)
        LIM = 1.6
        out = sum(int((np.linalg.norm(np.array(r["spectrum"][key]), axis=1)
                       > LIM).sum()) for r in rows)
        extra = f"\n{out} beyond frame" if out else ""
        ax.set_title(f"{ttl}\nmax |$\\lambda$| = {rad:.2f}{extra}", fontsize=6.5)
        ax.set_xlim(-LIM, LIM); ax.set_ylim(-LIM, LIM)
        ax.set_aspect("equal"); ax.grid(False)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("Im $\\lambda$", fontsize=7)
    for ax in axes:
        ax.set_xlabel("Re $\\lambda$", fontsize=7)
    _save(fig, "fig4_two_properties", outdir)


# ------------------------------------------------------------------ figure 5
def fig5(res, outdir):
    """Parametric collapse: usable horizon against Reynolds number."""
    rows = [r for r in _load(f"{res}/parwake_v2/seed*.json")
            if r.get("delta") is not None]
    if not rows:
        print("  skipping fig5: no parwake_v2 rows")
        return
    kinds = sorted({r["kind"] for r in rows})
    cols = [C["skyblue"], C["vermillion"], C["green"], C["purple"], C["black"]]

    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    for (k, col, mk) in zip(kinds, cols, MARKERS):
        sub = [r for r in rows if r["kind"] == k]
        Re = np.array(sorted({r["Re"] for r in sub}))
        hz = [np.median([1 / (4 * abs(r["delta"])) for r in sub if r["Re"] == R])
              for R in Re]
        ax.plot(Re, hz, marker=mk, ms=4, color=col, label=k,
                markeredgecolor="white", markeredgewidth=0.4)
    tr = sorted({r["Re"] for r in rows if r["split"] == "train"})
    for R in tr:
        ax.axvline(R, color="#e8e8e8", lw=3, zorder=0)
    ax.annotate("shaded: training Re", (0.98, 0.03), xycoords="axes fraction",
                fontsize=6, color=MUTED, ha="right", va="bottom")
    ax.set_yscale("log")
    ax.set_xlabel("Reynolds number")
    ax.set_ylabel("usable horizon  $1/(4|\\delta|)$  (periods)")
    ax.set_title("Conditioning on a parameter costs the horizon")
    ax.legend(loc="best", fontsize=6.5)
    _save(fig, "fig5_parametric_collapse", outdir)


# ------------------------------------------------------------------ figure 6
def fig6(res, outdir):
    """Where delta is defined: regime map of the FNO benchmark's own equations.

    The benchmark ships three viscosities and all of them lie past the
    transition, so a period fit on any of them is ill-posed. The figure is the
    argument for why a new operating point was needed, and it doubles as a
    worked example of the split-half test: one horizontal line separates the
    cases where delta means something from the cases where it does not.
    """
    rows = []
    for sub in ("kolmogorov", "kolm_onset", "kolm_data"):
        for f in sorted(glob.glob(f"{res}/{sub}/*.json")):
            rows.extend(json.load(open(f)))
    if not rows:
        print("  skipping fig6: no sweep data")
        return
    best = {}
    for r in rows:                      # keep the longest record per viscosity
        k = round(r["nu"], 12)
        if k not in best or r.get("periods_in_record", 0) > best[k].get("periods_in_record", 0):
            best[k] = r
    rows = sorted(best.values(), key=lambda r: r["nu"])

    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    THRESH = 1e-7
    groups = {"steady": (C["skyblue"], "s", "steady"),
              "periodic": (C["green"], "o", "periodic"),
              "quasi-periodic or chaotic": (C["vermillion"], "^", "aperiodic")}
    for key, (col, mk, lab) in groups.items():
        sub = [r for r in rows if r["regime"] == key
               and r["split_half"] == r["split_half"]]
        if not sub:
            continue
        ax.scatter([r["nu"] for r in sub], [r["split_half"] for r in sub],
                   s=34, color=col, marker=mk, edgecolor="white", linewidth=0.6,
                   zorder=3, label=lab)
    ax.axhline(THRESH, color=INK, lw=1.0, ls=(0, (4, 3)), zorder=2)
    ax.annotate("well-posed period fit below this line", (1.5e-3, THRESH * 1.6),
                fontsize=6, color=INK, ha="left")
    for nu in (1e-3, 1e-4):
        ax.axvline(nu, color="#e3e3e3", lw=6, zorder=0)
    ax.annotate("FNO benchmark settings\n(a third, ν = 10⁻⁵, lies further left)",
                (8.5e-4, 5e0), fontsize=6, color=MUTED, ha="right", va="top")
    ax.annotate("steady: no oscillation to fit", (7e-3, 5e-3), fontsize=6,
                color=MUTED, ha="center")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("viscosity  ν")
    ax.set_ylabel("split-half instability of the period fit")
    ax.set_title("δ is undefined at every setting the benchmark ships", fontsize=8)
    ax.legend(loc="lower right", fontsize=6.5)
    _save(fig, "fig6_regime_map", outdir)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--outdir", default="figures")
    ap.add_argument("--only", type=int, nargs="*")
    a = ap.parse_args()
    w = _validate_palette()
    print(f"palette worst-case CVD dE = {w:.1f} (target >= 8)")
    assert w >= 8, "palette fails CVD separation"
    for i, fn in enumerate((fig1, fig2, fig3, fig4, fig5, fig6), start=1):
        if a.only and i not in a.only:
            continue
        print(f"figure {i}: {fn.__doc__.splitlines()[0]}")
        try:
            fn(a.results, a.outdir)
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
