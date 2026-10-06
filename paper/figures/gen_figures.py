"""Data figures for the paper. Run from the project root: python3 paper/figures/gen_figures.py
Reads raw run logs in pilot-logs/{r3,r4,r5}; writes PDFs next to this file."""
import glob, json, os
import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.dirname(os.path.abspath(__file__))
plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "font.family": "DejaVu Sans",
                     "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
                     "lines.linewidth": 1.3, "axes.linewidth": 0.6})
COL = {"full": "#222222", "rowonly": "#0072B2", "dimbal": "#E69F00", "scalar": "#D55E00"}
LAB = {"full": "Full Adam", "rowonly": "Row-level", "dimbal": "Column-level", "scalar": "Matrix-level"}
MK = {"full": "o", "rowonly": "s", "dimbal": "^", "scalar": "D"}


def load(prefix, dirs=("pilot-logs/r3", "pilot-logs/r4", "pilot-logs/r5")):
    r = {}
    for d in dirs:
        for f in sorted(glob.glob(f"{d}/{prefix}_s*.json")):
            r[int(f.rsplit("_s", 1)[1][:-5])] = json.load(open(f))
    return r


def val(l, k):
    if k == "auc":
        return float(np.mean([s["rankme"] for s in l["series"]]))
    if k in ("knn_acc", "probe_acc"):
        return 100 * l[k]
    return l["series"][-1][k]


def pdiff(a, b, k):
    s = sorted(set(a) & set(b))
    d = np.array([val(a[i], k) - val(b[i], k) for i in s])
    h = stats.t.ppf(0.975, len(d) - 1) * d.std(ddof=1) / np.sqrt(len(d))
    return d.mean(), h


def fig_traj():
    fig, ax = plt.subplots(1, 4, figsize=(7.2, 1.95), sharey=True)
    k = 0
    for c, cn in [("0", "no regularizer"), ("0.03", "pooled VICReg 0.03")]:
        for v, vn in [("mat", "matched"), ("nat", "unmatched")]:
            a = ax[k]; k += 1
            for o in ["full", "rowonly", "dimbal", "scalar"]:
                r = load(f"p4_full_c{c}" if o == "full" else f"p4_{o}_{v}_c{c}", ("pilot-logs/r5",))
                st = [s["step"] for s in next(iter(r.values()))["series"]]
                y = np.array([[s["rankme"] for s in l["series"]] for l in r.values()])
                m, sd = y.mean(0), y.std(0, ddof=1)
                a.plot(np.array(st) / 1000, m, color=COL[o], label=LAB[o])
                a.fill_between(np.array(st) / 1000, m - sd, m + sd, color=COL[o], alpha=0.15, lw=0)
            a.set_title(f"{cn}\n{vn}"); a.set_xlabel("training step (×1000)")
    ax[0].set_ylabel("RankMe"); ax[3].legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.4); fig.savefig(f"{OUT}/fig2_trajectories.pdf"); plt.close(fig)


def fig_forest():
    S = [("CIFAR-100, no reg.", "p4_full_c0", "p4_{}_mat_c0"),
         ("CIFAR-100, pooled VICReg", "p4_full_c0.03", "p4_{}_mat_c0.03"),
         ("CIFAR-100, projector VICReg", "rv_proj_full", "rv_proj_{}_mat"),
         ("CIFAR-100, SIGReg", "rv_sigreg_full", "rv_sigreg_{}_mat"),
         ("STL-10, no reg.", "stl_full_c0", "stl_{}_mat_c0"),
         ("STL-10, pooled VICReg", "stl_full_c0.03", "stl_{}_mat_c0.03")]
    M = [("auc", "Δ RankMe, trajectory mean"), ("knn_acc", "Δ kNN accuracy (points)"), ("probe_acc", "Δ linear probe (points)")]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.7), sharey=True)
    for j, (k, kn) in enumerate(M):
        a = ax[j]
        for i, (sn, fp, tp) in enumerate(S):
            full = load(fp, ("pilot-logs/r5",))
            for q, o in enumerate(["rowonly", "dimbal", "scalar"]):
                m, h = pdiff(load(tp.format(o), ("pilot-logs/r5",)), full, k)
                y = -i + (1 - q) * 0.24
                a.errorbar(m, y, xerr=h, fmt=MK[o], color=COL[o], ms=3.2, lw=0.9, capsize=1.5,
                           label=LAB[o] if i == 0 else None)
        a.axvline(0, color="#888888", lw=0.6, ls="--"); a.set_xlabel(kn)
        a.set_yticks([-i for i in range(len(S))]); a.set_yticklabels([s[0] for s in S])
        a.tick_params(axis="y", length=0)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.93))
    h, l = ax[0].get_legend_handles_labels()
    fig.legend(h, l, frameon=False, loc="upper center", ncol=3, handletextpad=0.2)
    fig.savefig(f"{OUT}/fig3_paired_differences.pdf"); plt.close(fig)


def fig_surface():
    CS = ["0", "0.01", "0.03", "0.1", "0.3", "1"]
    fig, ax = plt.subplots(1, 3, figsize=(7.2, 2.0))
    for j, (k, kn) in enumerate([("rankme", "RankMe"), ("knn_acc", "kNN accuracy (%)"), ("probe_acc", "linear probe (%)")]):
        for o in ["full", "rowonly", "dimbal", "scalar"]:
            m, sd = [], []
            for c in CS:
                x = np.array([val(l, k) for l in load(f"reg_{o}_c{c}", ("pilot-logs/r3", "pilot-logs/r4")).values()])
                m.append(x.mean()); sd.append(x.std(ddof=1))
            ax[j].errorbar(range(len(CS)), m, yerr=sd, color=COL[o], marker=MK[o], ms=3, capsize=1.5, lw=1.1, label=LAB[o])
        ax[j].set_xticks(range(len(CS))); ax[j].set_xticklabels(CS)
        ax[j].set_xlabel("pooled VICReg coefficient"); ax[j].set_ylabel(kn)
    ax[0].legend(frameon=False, loc="lower right")
    fig.tight_layout(pad=0.4); fig.savefig(f"{OUT}/fig4_response_surface.pdf"); plt.close(fig)


def fig_surgery():
    ff, dd = load("n2r3_shamff_600", ("pilot-logs/r3",)), load("n2r3_shamdd_600", ("pilot-logs/r3",))
    A = [("full → column, step 300", "n2r3_fwd_300", ff, "dimbal"), ("full → column, step 600", "n2r3_fwd_600", ff, "dimbal"),
         ("full → column, step 1000", "n2r3_fwd_1000", ff, "dimbal"),
         ("full + step-size profile only", "n2r3_pertensor_600", ff, "full"),
         ("full + last block lr × 1.2", "n2r3_lastblk_600", ff, "full"),
         ("full → matrix, step 600", "n2r3_fwd_scalar_600", ff, "scalar"), ("full → row, step 600", "n2r3_fwd_row_600", ff, "rowonly"),
         ("column → row, step 600", "n2r3_rev_row_600", dd, "rowonly"), ("column → matrix, step 600", "n2r3_rev_scalar_600", dd, "scalar"),
         ("column → full, step 600", "n2r3_rev_full_600", dd, "full")]
    fig, a = plt.subplots(figsize=(3.5, 2.5))
    for i, (n, p, ref, o) in enumerate(A):
        m, h = pdiff(load(p, ("pilot-logs/r3",)), ref, "rankme")
        a.errorbar(m, -i, xerr=h, fmt=MK[o], color=COL[o], ms=3.5, lw=0.9, capsize=1.5)
    a.axvline(0, color="#888888", lw=0.6, ls="--"); a.axhline(-6.5, color="#cccccc", lw=0.5)
    a.set_yticks([-i for i in range(len(A))]); a.set_yticklabels([x[0] for x in A]); a.tick_params(axis="y", length=0)
    a.set_xlabel("Δ final RankMe vs. sham switch")
    fig.tight_layout(pad=0.4); fig.savefig(f"{OUT}/fig5_surgery.pdf"); plt.close(fig)


if __name__ == "__main__":
    fig_traj(); fig_forest(); fig_surface(); fig_surgery()
    print("ok")
