"""Build update-RMS reference trajectories by averaging runs over seeds.
Usage: python make_rms_ref.py 'pilot-logs/x_s*.json' out.json            (global, dense if available)
       python make_rms_ref.py --tensor 'pilot-logs/x_s*.json' out.npz    (per-parameter, from *_dense.npz)
The output is the counterfactual RMS the (exact or lagged) matching controller tracks."""
import sys, glob, json
import numpy as np

tensor = sys.argv[1] == "--tensor"
pattern, out = sys.argv[-2], sys.argv[-1]
fs = sorted(glob.glob(pattern))
assert fs, f"no files match {pattern}"
if tensor:
    zs = [np.load(f.replace(".json", "_dense.npz")) for f in fs]
    names = zs[0]["names"]
    assert all(list(z["names"]) == list(names) for z in zs)
    rms = np.mean([z["rms"] for z in zs], axis=0)
    np.savez_compressed(out, names=names, rms=rms.astype(np.float32))
    print(f"wrote {out}: {rms.shape} from {len(fs)} runs", flush=True)
    sys.exit(0)
logs = [json.load(open(f)) for f in fs]
if all(l.get("dense", {}).get("update_rms") for l in logs):          # R3: dense per-step reference
    arr = np.array([l["dense"]["update_rms"] for l in logs], dtype=float)
    steps = list(range(1, arr.shape[1] + 1)); urms = arr.mean(0).tolist()
else:                                                                # legacy: eval-cadence points
    per_step = {}
    for l in logs:
        for s in l["series"]:
            u = s.get("update_rms")
            if u and u == u and u > 0:
                per_step.setdefault(s["step"], []).append(u)
    steps = sorted(per_step); urms = [float(np.mean(per_step[st])) for st in steps]
json.dump({"step": steps, "update_rms": urms, "n_runs": len(fs), "src": pattern}, open(out, "w"))
print(f"wrote {out}: {len(steps)} steps from {len(fs)} runs, uRMS {urms[0]:.3e}..{urms[-1]:.3e}", flush=True)
