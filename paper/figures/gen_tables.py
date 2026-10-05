"""LaTeX tables from raw run logs. Run from the project root: python3 paper/figures/gen_tables.py"""
import glob, json, os
import numpy as np
from scipy import stats

D = "pilot-logs/r5"
OUT = "paper/tables"
NAME = {"full": "Full Adam", "rowonly": "Row-level", "dimbal": "Column-level", "scalar": "Matrix-level"}


def load(prefix, seeds=None):
    r = {}
    for f in sorted(glob.glob(f"{D}/{prefix}_s*.json")):
        s = int(f.rsplit("_s", 1)[1][:-5])
        if seeds is None or s in seeds:
            r[s] = json.load(open(f))
    return r


def val(l, k):
    if k == "auc":
        return float(np.mean([s["rankme"] for s in l["series"]]))
    if k in ("knn_acc", "probe_acc"):
        return 100 * l[k]
    if k == "few5":
        return 100 * l["battery"]["fewshot_5"]
    if k == "mb":
        return l["opt_state_bytes"] / 2 ** 20
    if k == "scale":
        sc = l["dense"].get("scale")
        return float(np.mean(sc[200:])) if sc else float("nan")
    return l["series"][-1][k]


def fmt(x, d=1, sign=False):
    """Round half up (as in the analysis summaries), after removing float noise."""
    from decimal import Decimal, ROUND_HALF_UP
    q = Decimal(repr(round(float(x), 8))).quantize(Decimal(1).scaleb(-d), rounding=ROUND_HALF_UP)
    if q == 0: q = abs(q)
    return f"{q:+f}" if sign else f"{q:f}"


def ms(r, k, d=1):
    x = np.array([val(l, k) for l in r.values()])
    return f"{fmt(x.mean(), d)} $\\pm$ {fmt(x.std(ddof=1), d)}"


def mean(r, k, d=1):
    return fmt(np.mean([val(l, k) for l in r.values()]), d)


def diff(a, b, k, d=2):
    s = sorted(set(a) & set(b))
    x = np.array([val(a[i], k) - val(b[i], k) for i in s])
    h = stats.t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    lo, hi = x.mean() - h, x.mean() + h
    txt = f"${fmt(x.mean(), d, True)}$ [${fmt(lo, d, True)}$, ${fmt(hi, d, True)}$]"
    return txt + ("$^{*}$" if lo > 0 or hi < 0 else "")


def write(name, lines):
    open(f"{OUT}/{name}.tex", "w").write("\n".join(lines) + "\n")


def tab_main():
    L = ["\\begin{tabular}{llrrrrrrr}", "\\toprule",
         "$c$ & Second moment & State & RankMe & erank & Traj.\\ mean & kNN & Probe & 5-shot \\\\", "\\midrule"]
    for c in ["0", "0.03"]:
        rows = [("Full Adam", f"p4_full_c{c}")]
        for v, vn in [("mat", "matched"), ("nat", "unmatched")]:
            rows += [(f"{NAME[o]}, {vn}", f"p4_{o}_{v}_c{c}") for o in ["rowonly", "dimbal", "scalar"]]
        for i, (n, p) in enumerate(rows):
            r = load(p)
            L.append(f"{c if i == 0 else ''} & {n} & {mean(r, 'mb')} & {ms(r, 'rankme')} & {ms(r, 'erank')} & {mean(r, 'auc')} & "
                     f"{ms(r, 'knn_acc')} & {ms(r, 'probe_acc')} & {mean(r, 'few5')} \\\\")
        if c == "0":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("tab_main", L)


def tab_main_diff():
    L = ["\\begin{tabular}{lllll}", "\\toprule",
         "$c$ & Second moment & $\\Delta$ RankMe & $\\Delta$ traj.\\ mean & $\\Delta$ kNN & $\\Delta$ probe \\\\".replace("lllll", "llllll"), "\\midrule"]
    L[0] = "\\begin{tabular}{llllll}"
    for c in ["0", "0.03"]:
        full = load(f"p4_full_c{c}"); i = 0
        for v, vn in [("mat", "matched"), ("nat", "unmatched")]:
            for o in ["rowonly", "dimbal", "scalar"]:
                r = load(f"p4_{o}_{v}_c{c}")
                L.append(f"{c if i == 0 else ''} & {NAME[o]}, {vn} & {diff(r, full, 'rankme')} & {diff(r, full, 'auc')} & "
                         f"{diff(r, full, 'knn_acc')} & {diff(r, full, 'probe_acc')} \\\\"); i += 1
        if c == "0":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("tab_main_diff", L)


def tab_opt():
    O = [("adam8bit", "8-bit Adam"), ("q8v", "$v$-only 8-bit"), ("adafactor", "Factored-$v$ AdamW"),
         ("rowperm", "Row-permuted control"), ("minipart", "Adam-mini partition"), ("adammini", "Adam-mini (fallback)"), ("galore", "GaLore")]
    L = ["\\begin{tabular}{lllrrlll}", "\\toprule",
         "$c$ & Optimizer & Protocol & State & Scale & $\\Delta$ RankMe & $\\Delta$ kNN & $\\Delta$ probe \\\\", "\\midrule"]
    for c in ["0", "0.03"]:
        full = load(f"p4_full_c{c}", (10, 11, 12)); i = 0
        for o, n in O:
            for v, vn in [("mat", "matched"), ("nat", "unmatched")]:
                r = load(f"r5_{o}_{v}_c{c}")
                if not r:
                    continue
                sc = mean(r, "scale", 2) if v == "mat" else "--"
                L.append(f"{c if i == 0 else ''} & {n} & {vn} & {mean(r, 'mb')} & {sc} & {diff(r, full, 'rankme', 1)} & "
                         f"{diff(r, full, 'knn_acc')} & {diff(r, full, 'probe_acc')} \\\\"); i += 1
        if c == "0":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("tab_opt", L)


def tab_other():
    S = [("CIFAR-100, projector VICReg", "rv_proj_full", "rv_proj_{}_mat"), ("CIFAR-100, SIGReg", "rv_sigreg_full", "rv_sigreg_{}_mat"),
         ("STL-10, no regularizer", "stl_full_c0", "stl_{}_mat_c0"), ("STL-10, pooled VICReg 0.03", "stl_full_c0.03", "stl_{}_mat_c0.03")]
    L = ["\\begin{tabular}{llrrrlll}", "\\toprule",
         "Setting & Second moment & RankMe & kNN & Probe & $\\Delta$ RankMe & $\\Delta$ kNN & $\\Delta$ probe \\\\", "\\midrule"]
    for j, (sn, fp, tp) in enumerate(S):
        full = load(fp)
        L.append(f"{sn} & Full Adam & {ms(full, 'rankme')} & {ms(full, 'knn_acc')} & {ms(full, 'probe_acc')} & -- & -- & -- \\\\")
        for o in ["rowonly", "dimbal", "scalar"]:
            r = load(tp.format(o))
            L.append(f" & {NAME[o]} & {ms(r, 'rankme')} & {ms(r, 'knn_acc')} & {ms(r, 'probe_acc')} & {diff(r, full, 'rankme', 1)} & "
                     f"{diff(r, full, 'knn_acc')} & {diff(r, full, 'probe_acc')} \\\\")
        if j < len(S) - 1:
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}"]
    write("tab_other", L)


if __name__ == "__main__":
    tab_main(); tab_main_diff(); tab_opt(); tab_other()
    for f in sorted(glob.glob(f"{OUT}/*.tex")):
        print("==", f); print(open(f).read())
