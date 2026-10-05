"""Stage-2 trainer for the RowScaleAdam / JEPA optimizer-collapse study.

Trains a small I-JEPA and logs collapse diagnostics + kNN/linear-probe over training.
Adds (vs pilot): dataset selection, update-RMS logging (confound control), --lr_scale for
update-norm matching, general moment surgery (forward & reverse), linear probe.
"""
import argparse, json, os, time
import numpy as np
import torch
from data import CIFAR100Memory, STL10Memory
from model import IJEPA, sample_masks
from metrics import collapse_stats, rankme
from optim import AdamFamily, ExternalMatched, adam_mini_partition

ADAM_MODES = ("full", "scalar", "dimbal", "rowonly", "adafactor", "q8v", "rowperm", "minipart")


def build_optimizer(model, name, lr, wd):
    mods = [model.encoder, model.predictor] + ([model.projector] if model.projector is not None else [])
    params = [p for m in mods for p in m.parameters()]
    if name in ADAM_MODES:
        opt = AdamFamily(params, lr=lr, weight_decay=wd, mode=name)
        if name == "minipart":   # Adam-mini's intended partition inside the controlled family
            named = [(f"{i}.{n}", p) for i, m in enumerate(mods) for n, p in m.named_parameters()]
            attn = {f"{i}.{n}.in_proj_weight": a for i, m in enumerate(mods) for n, a in m.named_modules()
                    if isinstance(a, torch.nn.MultiheadAttention)}
            opt.part = adam_mini_partition(named, lambda n: attn[n].embed_dim, lambda n: attn[n].num_heads)
        return opt
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True)
    if name == "adam8bit":   # real memory-efficient optimizer: quantizes per-coordinate moments
        import bitsandbytes as bnb
        return ExternalMatched(bnb.optim.Adam8bit(params, lr=lr, weight_decay=0.0), wd, nosync=True)
    if name == "galore":     # low-rank GRADIENT projection (orthogonal axis to v geometry)
        from galore_torch import GaLoreAdamW
        mats = [p for p in params if p.dim() == 2]
        rest = [p for p in params if p.dim() != 2]
        groups = [{"params": mats, "rank": 64, "update_proj_gap": 200, "scale": 0.25, "proj_type": "std"},
                  {"params": rest}]
        return ExternalMatched(GaLoreAdamW(groups, lr=lr, weight_decay=0.0, no_deprecation_warning=True), wd)
    if name == "adammini":   # block-wise v (fewer learning rates); per-neuron for unknown layers
        from adam_mini import Adam_mini
        named = [(f"{i}.{n}", p) for i, m in enumerate(mods) for n, p in m.named_parameters()]
        return ExternalMatched(Adam_mini(named_parameters=named, lr=lr, weight_decay=0.0, dim=model.dim,
                                         n_heads=model.encoder.blocks[-1].attn.num_heads), wd)
    raise ValueError(name)


def opt_state_bytes(opt):
    if hasattr(opt, "optimizer_state_bytes"):
        return opt.optimizer_state_bytes()
    tot = 0
    for st in opt.state.values():
        for k, v in (st.items() if isinstance(st, dict) else []):
            if torch.is_tensor(v) and k != "params" and not isinstance(v, torch.nn.Parameter):
                tot += v.numel() * v.element_size()
            elif hasattr(v, "ortho_matrix") and torch.is_tensor(v.ortho_matrix):   # GaLore projector
                tot += v.ortho_matrix.numel() * v.ortho_matrix.element_size()
    return tot


def build_data(dataset, dev, subset):
    if dataset == "cifar100":
        ssl = CIFAR100Memory("data", "train", dev, subset=subset)
        bank = ssl
        query = CIFAR100Memory("data", "test", dev)
        return ssl, bank, query, 32, 4
    if dataset == "stl10":
        ssl = STL10Memory("data", "unlabeled", dev, subset=subset)
        bank = STL10Memory("data", "train", dev)
        query = STL10Memory("data", "test", dev)
        return ssl, bank, query, 96, 8
    if dataset == "in100":
        from data_in100 import IN100Memory
        ssl = IN100Memory(None, "ssl", dev, subset=subset)  # labeled non-query; also serves as kNN bank
        query = IN100Memory(None, "query", dev)
        return ssl, ssl, query, 128, 16
    raise ValueError(dataset)


def repr_facing_params(model):
    """Matrices whose output channel == residual/embedding channel (cleanest interpretation)."""
    blk = model.encoder.blocks[-1]
    d = {"fc2_last": blk.fc2.weight, "attn_out_last": blk.attn.out_proj.weight,
         "patch_embed": model.encoder.proj.weight}
    if getattr(model.predictor, "depth", 0) > 0:
        d["pred_out"] = model.predictor.out.weight
    return d


def v_per_channel(opt, p):
    st = opt.state.get(p, {})
    if "v" not in st or st["v"].dim() < 2:
        return None
    return st["v"].mean(dim=1).detach().cpu()


def switch_surgery(opt, target):
    """Moment surgery (see AdamFamily.switch_geometry). R3 fix: the source geometry is read from the
    recorded st["vmode"], not inferred from v.numel() (which misread dimbal as rowonly on square
    matrices such as attn.out_proj)."""
    opt.switch_geometry(target)


def linear_probe(zb, yb, zq, yq, nclass):
    """Quick logistic-regression linear probe on frozen embeddings."""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Xb = zb.cpu().numpy(); Xq = zq.cpu().numpy()
        sc = StandardScaler().fit(Xb)
        clf = LogisticRegression(max_iter=300, C=1.0)
        clf.fit(sc.transform(Xb), yb.cpu().numpy())
        return float((clf.predict(sc.transform(Xq)) == yq.cpu().numpy()).mean())
    except Exception as e:
        print("linear_probe skipped:", e, flush=True)
        return float("nan")


def downstream_battery(zb, yb, zq, yq, seed=0):
    """Rank-causality falsification battery (Codex 2026-06-01): does the embedding's downstream
    quality track its effective-rank across interventions? Beyond kNN-20 we add a LINEAR probe and
    FEW-SHOT logistic probes (1/5/10 labels per class), all on frozen embeddings. If rank moves a
    lot via optimizer geometry but these stay flat, rank is downstream-irrelevant (the paper's thesis)."""
    import numpy as np
    out = {}
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        Xb = zb.cpu().numpy(); Yb = yb.cpu().numpy()
        Xq = zq.cpu().numpy(); Yq = yq.cpu().numpy()
        sc = StandardScaler().fit(Xb)
        Xb_s, Xq_s = sc.transform(Xb), sc.transform(Xq)
        classes = np.unique(Yb)
        # full linear probe (all bank labels)
        clf = LogisticRegression(max_iter=300, C=1.0).fit(Xb_s, Yb)
        out["probe_full"] = float((clf.predict(Xq_s) == Yq).mean())
        # few-shot: k labeled examples per class, averaged over 3 sampling repeats
        rng = np.random.RandomState(1234 + seed)
        for k in (1, 5, 10):
            accs = []
            for _ in range(3):
                idx = []
                for c in classes:
                    ci = np.where(Yb == c)[0]
                    if len(ci) == 0:
                        continue
                    idx.extend(rng.choice(ci, size=min(k, len(ci)), replace=False))
                idx = np.array(idx)
                cf = LogisticRegression(max_iter=300, C=1.0).fit(Xb_s[idx], Yb[idx])
                accs.append((cf.predict(Xq_s) == Yq).mean())
            out[f"fewshot_{k}"] = float(np.mean(accs))
    except Exception as e:
        print("downstream_battery skipped:", e, flush=True)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--optimizer", default="full",
                    choices=list(ADAM_MODES) + ["sgd", "adam8bit", "galore", "adammini"])
    ap.add_argument("--dataset", default="cifar100", choices=["cifar100", "stl10", "in100"])
    ap.add_argument("--lr", type=float, default=1.5e-3)
    ap.add_argument("--lr_scale", type=float, default=1.0)  # multiplier for update-RMS matching
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--ema", type=float, default=0.996)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--bs", type=int, default=256)
    ap.add_argument("--dim", type=int, default=192)
    ap.add_argument("--depth", type=int, default=12)
    ap.add_argument("--heads", type=int, default=3)
    ap.add_argument("--pred_depth", type=int, default=2)
    ap.add_argument("--vicreg_coef", type=float, default=0.0)  # >0: add VICReg anti-collapse reg (MA3)
    # R5: regularizer family. pooled = VICReg var+cov on the pooled context embedding (R3/R4 default);
    # proj = standard VICReg var + cov/25 on a 2-layer projector output; sigreg = LeJEPA sketched
    # isotropic-Gaussian (Epps-Pulley) regularizer on the pooled context embedding.
    ap.add_argument("--reg_type", default="pooled", choices=["pooled", "proj", "sigreg"])
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--eval_every", type=int, default=100)
    ap.add_argument("--eval_n", type=int, default=2000)
    ap.add_argument("--n_target", type=int, default=4)
    ap.add_argument("--surgery_step", type=int, default=-1)
    ap.add_argument("--surgery_to", default="dimbal", choices=["dimbal", "rowonly", "full", "scalar"])
    # update-RMS matching across the surgery boundary (ARS M4): after surgery, rescale lr each step
    # so realized update-RMS tracks the pre-surgery (source-geometry) RMS -> isolates the axis change
    # from a step-size confound. Controller: surgery_mult *= rms_target / last_update_rms (clamped).
    ap.add_argument("--surgery_match", action="store_true")
    ap.add_argument("--surgery_match_window", type=int, default=50)
    # match target: 'snapshot' = freeze the pre-surgery RMS (the value at surgery_step); 'trajectory' =
    # track the CONCURRENT geometry-preserving (sham-full) per-step RMS loaded from --rms_ref_file, so the
    # surgery run is held step-for-step at the rank-preserving counterfactual's update-RMS (N2: the strongest
    # step-size-confound control). EMA-smooth the realized RMS in the controller to suppress per-step noise.
    ap.add_argument("--surgery_match_mode", default="snapshot", choices=["snapshot", "trajectory"])
    ap.add_argument("--rms_ref_file", default="")        # json: {"step": [...], "update_rms": [...]}
    ap.add_argument("--surgery_match_ema", type=float, default=0.3)  # EMA alpha on realized RMS (0=off)
    # R3 controls -------------------------------------------------------------------------------
    # exact (same-step, non-lagged) GLOBAL update-RMS match to a dense per-step reference trajectory
    ap.add_argument("--exact_ref", default="")           # json: {"step": [...], "update_rms": [...]} dense
    ap.add_argument("--exact_from", type=int, default=0) # first matched step (0 -> surgery_step+1, or 1)
    # exact per-parameter update-RMS match (layerwise step-size mediation control): npz names/steps/rms
    ap.add_argument("--tensor_ref", default="")
    ap.add_argument("--tensor_from", type=int, default=601)
    # crude mediation control: multiply the lr of the last encoder block by this from --tensor_from on
    ap.add_argument("--lastblock_mult", type=float, default=1.0)
    # keep decoupled weight decay on the BASE schedule (lr * warmup), unaffected by lr_scale /
    # surgery_mult / exact matching, so step-size matching does not also rescale the decay
    ap.add_argument("--wd_decouple", action="store_true")
    # healthy-recipe options (Gate H): cosine lr decay after warmup, EMA momentum ramp ema -> ema_end,
    # and a held-out VALIDATION query (last 5000 train images) for recipe selection instead of test
    ap.add_argument("--cosine", action="store_true")
    ap.add_argument("--ema_end", type=float, default=0.0)   # 0 -> constant --ema
    ap.add_argument("--val_query", action="store_true")
    ap.add_argument("--dense_log", action="store_true")  # per-step per-parameter update-RMS -> *_dense.npz
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--subset", type=int, default=None)
    ap.add_argument("--ref_step", type=int, default=500)
    ap.add_argument("--bank_n", type=int, default=10000)
    ap.add_argument("--query_n", type=int, default=5000)
    ap.add_argument("--probe", action="store_true")
    ap.add_argument("--battery", action="store_true")  # rank-causality downstream battery (probe + few-shot)
    ap.add_argument("--knn_every", type=int, default=0)  # periodic kNN trajectory (0=final only)
    ap.add_argument("--tag", default="run")
    ap.add_argument("--out", default="pilot-logs/run.json")
    args = ap.parse_args()

    torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
    dev = "cuda"
    ssl, bank, query, img, patch = build_data(args.dataset, dev, args.subset)
    if args.val_query:
        assert args.dataset == "cifar100" and args.bank_n <= 45000
        class _Val:                                     # last 5000 train images, labels used only for eval
            def eval_batch(self, n):
                return ssl.x[-5000:][:n], ssl.y[-5000:][:n]
        query = _Val()
    model = IJEPA(img=img, patch=patch, dim=args.dim, depth=args.depth, heads=args.heads,
                  pred_depth=args.pred_depth, ema=args.ema, vicreg_coef=args.vicreg_coef,
                  reg_type=args.reg_type).to(dev)
    opt = build_optimizer(model, args.optimizer, args.lr, args.wd)
    base_lr = args.lr * args.lr_scale
    rfp = repr_facing_params(model)
    eval_x, _ = ssl.eval_batch(args.eval_n) if hasattr(ssl, "x") else (bank.eval_batch(args.eval_n)[0], None)
    # for SSL split without labels (STL unlabeled), use bank (labeled train) imgs for diagnostics
    if args.dataset == "stl10":
        eval_x = bank.eval_batch(args.eval_n)[0]

    # preload bank/query for periodic kNN trajectory
    knn_bank = knn_query = None
    if args.knn_every > 0:
        bx0, by0 = bank.eval_batch(args.bank_n); qx0, qy0 = query.eval_batch(args.query_n)
        knn_bank = (bx0, by0); knn_query = (qx0, qy0)

    def knn_now():
        zb = torch.nn.functional.normalize(model.embed(knn_bank[0]), dim=1)
        zq = torch.nn.functional.normalize(model.embed(knn_query[0]), dim=1)
        pred = torch.mode(knn_bank[1][(zq @ zb.T).topk(20, dim=1).indices], dim=1).values
        return (pred == knn_query[1]).float().mean().item()

    log = {"args": vars(args), "series": [], "v_ref": None}
    ref_v = {}
    t0 = time.time()
    upd_rms_acc, tot_rms_acc, upd_rms_n = 0.0, 0.0, 0
    surgery_mult = 1.0          # post-surgery lr multiplier (update-RMS matching controller)
    rms_window = []             # pre-surgery realized update-RMS (source geometry), for the match target
    rms_target = None
    rms_realized_ema = None     # EMA of realized update-RMS for the controller (suppresses per-step noise)
    # trajectory match: load the concurrent geometry-preserving (sham-full) per-step RMS reference
    ref_steps, ref_urms = None, None
    if args.surgery_match and args.surgery_match_mode == "trajectory" and args.rms_ref_file:
        with open(args.rms_ref_file) as f:
            _ref = json.load(f)
        ref_steps = np.asarray(_ref["step"], dtype=float)
        ref_urms = np.asarray(_ref["update_rms"], dtype=float)
        print(f"[{args.tag}] trajectory match: loaded {len(ref_steps)}-pt RMS ref from {args.rms_ref_file}", flush=True)

    exact_steps = exact_urms = None
    if args.exact_ref:
        with open(args.exact_ref) as f:
            _ref = json.load(f)
        exact_steps = np.asarray(_ref["step"], dtype=float); exact_urms = np.asarray(_ref["update_rms"], dtype=float)
        assert len(exact_steps) >= args.steps, "exact_ref must be a DENSE (per-step) reference"
    exact_from = args.exact_from or (args.surgery_step + 1 if args.surgery_step > 0 else 1)
    named = [(f"{mn}.{n}", p) for mn, m in (("encoder", model.encoder), ("predictor", model.predictor))
             + ((("projector", model.projector),) if model.projector is not None else ())
             for n, p in m.named_parameters()]
    tref = None
    if args.tensor_ref:
        _t = np.load(args.tensor_ref, allow_pickle=False)
        assert list(_t["names"]) == [n for n, _ in named], "tensor_ref parameter names mismatch"
        tref = _t["rms"]                                  # (steps, n_params), row i <-> step i+1
    last_ids = {id(p) for n, p in named if n.startswith(f"encoder.blocks.{args.depth - 1}.")}
    dense = {"update_rms": [], "scale": [], "loss": []}
    dense_param = [] if args.dense_log else None

    def match_target(step):
        """RMS the controller should drive the post-surgery realized update-RMS toward."""
        if ref_steps is not None:
            return float(np.interp(step, ref_steps, ref_urms))   # concurrent sham-full trajectory
        return rms_target                                        # frozen pre-surgery snapshot

    for step in range(1, args.steps + 1):
        sched = min(1.0, step / max(1, args.warmup))
        prog = max(0.0, (step - args.warmup) / max(1, args.steps - args.warmup))
        if args.cosine:
            sched *= 0.5 * (1 + np.cos(np.pi * prog))
        ema_m = args.ema if not args.ema_end else args.ema_end - (args.ema_end - args.ema) * 0.5 * (1 + np.cos(np.pi * step / args.steps))
        lr = base_lr * sched * surgery_mult
        for g in opt.param_groups:
            g["lr"] = lr
            if args.wd_decouple:
                g["wd_lr"] = args.lr * sched
            if args.lastblock_mult != 1.0 and step >= args.tensor_from:
                g["tensor_lr_mult"] = {i: args.lastblock_mult for i in last_ids}
        if exact_steps is not None:
            opt.exact_rms_target = float(np.interp(step, exact_steps, exact_urms)) if step >= exact_from else None
        if tref is not None:
            opt.exact_tensor_rms = ({id(p): float(tref[step - 1, i]) for i, (_, p) in enumerate(named)
                                     if tref[step - 1, i] > 0} if step >= args.tensor_from else None)
        imgs, _ = ssl.sample_batch(args.bs, augment=True)
        ctx_idx, tgt_idx = sample_masks(args.bs, model.grid, n_target=args.n_target, device=dev)
        ctx_idx = ctx_idx.expand(args.bs, -1); tgt_idx = tgt_idx.expand(args.bs, -1)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = model(imgs, ctx_idx, tgt_idx)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(
            [p for gp in opt.param_groups for p in gp["params"]], 1e9).item()
        opt.step(); model.ema_update(float(ema_m))
        if hasattr(opt, "last_param_rms"):
            dense["update_rms"].append(opt.last_update_rms); dense["scale"].append(getattr(opt, "last_scale", 1.0))
            dense["loss"].append(float(loss.item()))
            if dense_param is not None:
                dense_param.append([opt.last_param_rms.get(id(p), 0.0) for _, p in named])
        if hasattr(opt, "last_update_rms") and step > args.warmup:
            upd_rms_acc += opt.last_update_rms
            tot_rms_acc += getattr(opt, "last_total_rms", opt.last_update_rms); upd_rms_n += 1

        # update-RMS matching: collect the pre-surgery (source geometry) RMS, then after the switch
        # nudge surgery_mult so realized RMS tracks that target (raw RMS varies slowly -> stable).
        if args.surgery_match and args.surgery_step > 0 and hasattr(opt, "last_update_rms"):
            r = opt.last_update_rms
            if step <= args.surgery_step and step > args.warmup and r == r:  # pre-switch, finite
                rms_window.append(r)
                rms_window[:] = rms_window[-args.surgery_match_window:]
            elif step > args.surgery_step and r == r and r > 0:
                a = args.surgery_match_ema
                rms_realized_ema = r if rms_realized_ema is None else (1 - a) * rms_realized_ema + a * r
                tgt = match_target(step)
                if tgt:
                    ratio = min(4.0, max(0.25, tgt / rms_realized_ema))
                    surgery_mult = min(50.0, max(0.02, surgery_mult * ratio))

        if args.surgery_step > 0 and step == args.surgery_step:
            switch_surgery(opt, args.surgery_to)
            if args.surgery_match and rms_window:
                rms_target = sum(rms_window) / len(rms_window)
                print(f"[{args.tag}] surgery@{step} match target uRMS={rms_target:.3e}", flush=True)
        if step == args.ref_step and args.optimizer == "full":
            for name, p in rfp.items():
                vc = v_per_channel(opt, p)
                if vc is not None:
                    ref_v[name] = vc

        if step % args.eval_every == 0 or step == 1:
            model.eval()
            with torch.no_grad():
                z = model.embed(eval_x)
                z_tgt = model.target(eval_x).mean(dim=1)   # readout the VICReg term never touches
            model.train()
            cs = collapse_stats(z); rm = rankme(z)
            cs_t = collapse_stats(z_tgt); rm_t = rankme(z_tgt)
            tensor_rms = {name: opt.last_tensor_rms.get(id(p)) for name, p in rfp.items()} \
                if hasattr(opt, "last_tensor_rms") else {}
            chan_cv = None; efflr_cv = {}
            if hasattr(opt, "per_output_channel_scale"):
                sc = opt.per_output_channel_scale(rfp.get("fc2_last"))
                if sc is not None and sc.numel() > 1:
                    chan_cv = float(sc.std() / (sc.mean() + 1e-12))  # per-output-channel LR dispersion
                for name, p in rfp.items():                # R3: dispersion of the EFFECTIVE lr 1/(sqrt(vhat)+eps)
                    e = opt.per_output_channel_scale(p, eff_lr=True)
                    if e is not None and e.numel() > 1:
                        efflr_cv[name] = float(e.std() / (e.mean() + 1e-12))
            match_tgt = match_target(step) if (args.surgery_match and step > args.surgery_step) else None
            rec = {"step": step, "loss": float(loss.item()), "lr": lr, "grad_norm": gnorm,
                   "surgery_mult": surgery_mult, "match_target": match_tgt,
                   "rankme": rm, "update_rms": getattr(opt, "last_update_rms", float("nan")),
                   "total_rms": getattr(opt, "last_total_rms", float("nan")),
                   "tensor_rms": tensor_rms, "chan_scale_cv": chan_cv, "efflr_cv": efflr_cv,
                   "rankme_tgt": rm_t, "erank_norm_tgt": cs_t["erank_norm"],
                   "exact_scale": getattr(opt, "last_scale", 1.0),
                   "pred_loss": getattr(model, "last_pred", None), "reg_loss": getattr(model, "last_reg", None),
                   **{k: v for k, v in cs.items() if k != "std_vec"}}
            if args.knn_every > 0 and (step % args.knn_every == 0):
                model.eval()
                with torch.no_grad():
                    rec["knn"] = knn_now()
                model.train()
            log["series"].append(rec)
            print(f"[{args.tag}] step {step:5d} loss {loss.item():.4f} erank/D {cs['erank_norm']:.3f} "
                  f"rankme {rm:.2f} knn {rec.get('knn', float('nan'))} uRMS {rec['update_rms']:.2e}", flush=True)
            log["last_std_vec"] = cs["std_vec"].tolist()

    # downstream probes
    model.eval()
    with torch.no_grad():
        bx, by = bank.eval_batch(args.bank_n); qx, qy = query.eval_batch(args.query_n)
        zb = model.embed(bx); zq = model.embed(qx)
        zbn = torch.nn.functional.normalize(zb, dim=1); zqn = torch.nn.functional.normalize(zq, dim=1)
        knn = (zqn @ zbn.T).topk(20, dim=1).indices
        pred = torch.mode(by[knn], dim=1).values
        log["knn_acc"] = (pred == qy).float().mean().item()
        nclass = int(by.max().item()) + 1
        log["probe_acc"] = linear_probe(zb, by, zq, qy, nclass) if args.probe else None
        if args.battery:
            log["battery"] = downstream_battery(zb, by, zq, qy, seed=args.seed)
    model.train()

    if ref_v:
        log["v_ref"] = {k: v.tolist() for k, v in ref_v.items()}
    log["opt_state_bytes"] = opt_state_bytes(opt)
    log["mean_update_rms"] = (upd_rms_acc / upd_rms_n) if upd_rms_n else getattr(opt, "last_update_rms", float("nan"))
    log["mean_total_rms"] = (tot_rms_acc / upd_rms_n) if upd_rms_n else getattr(opt, "last_total_rms", float("nan"))
    log["wall_sec"] = time.time() - t0
    log["peak_mem_mb"] = torch.cuda.max_memory_allocated() / 1e6
    log["dense"] = dense
    log["env"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0), "code": "R5"}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    if dense_param is not None:
        np.savez_compressed(args.out.replace(".json", "_dense.npz"), names=np.array([n for n, _ in named]),
                            rms=np.asarray(dense_param, dtype=np.float32))
    with open(args.out, "w") as f:
        json.dump(log, f)
    print(f"[{args.tag}] DONE wall {log['wall_sec']:.1f}s peak {log['peak_mem_mb']:.0f}MB "
          f"kNN {log['knn_acc']:.4f} probe {log['probe_acc']} uRMS {log['mean_update_rms']} "
          f"opt_state {log['opt_state_bytes']} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
