"""Adam-family optimizer with a switchable second-moment geometry.

First moment m_t is ALWAYS per-coordinate; only the second-moment v_t geometry changes
(that is the lever under test). All modes use decoupled weight decay (AdamW).

For an (out,in) matrix, the second-moment state size and what it preserves:
  full      v[o,i]            -> out*in   (per-coordinate)           [costly baseline]
  adafactor v[o,i]=r_o*c_i    -> out+in   (rank-1: row AND col)      [memory-efficient, keeps output-channel scale]
  rowonly   v[o,i]=r_o        -> out      (per-OUTPUT-channel)       [METHOD: keeps output-channel scale]
  dimbal    v[o,i]=c_i        -> in       (per-INPUT, col-only)      [CONTROL: destroys output-channel scale]
  scalar    v[o,i]=s          -> 1        (single scalar)            [CONTROL: destroys all differentiation]

rowonly vs dimbal is the key controlled pair: ~equal memory (one vector each), but rowonly
preserves per-output-channel scale while dimbal flattens it. The hypothesis: rowonly resists
JEPA collapse, dimbal does not.

`last_update_rms` exposes RMS of the realized parameter update (for update-norm matching).
"""
import torch
from torch.optim import Optimizer

_MATRIX_MODES = ("scalar", "dimbal", "rowonly", "rowperm", "adafactor", "minipart")  # modes that special-case >=2D params
# R5 controls:
#   rowperm  rowonly statistics, but each row is divided by ANOTHER row's denominator (fixed random
#            permutation per tensor): same per-row state and same multiset of row scales, alignment destroyed.
#   q8v      per-coordinate v stored through a simulated block-wise 8-bit log quantizer (m stays fp32):
#            isolates second-moment precision from second-moment geometry.
Q8_BLOCK, Q8_MIN, Q8_DECADES = 2048, 4096, 7.0


def quantize8_(v):
    """In-place simulated 8-bit storage of a non-negative tensor: per-block absmax normalization, then 255
    log-spaced levels over Q8_DECADES decades (level 0 = exact zero). Relative error <= ~3.2%."""
    flat = v.reshape(-1)
    n = flat.numel(); pad = (-n) % Q8_BLOCK
    x = torch.nn.functional.pad(flat, (0, pad)).view(-1, Q8_BLOCK)
    amax = x.amax(dim=1, keepdim=True).clamp_min(1e-38)
    r = (x / amax).clamp_min(0)
    code = torch.round(254 * (torch.log10(r.clamp_min(1e-30)) + Q8_DECADES) / Q8_DECADES) + 1
    code = torch.where(r < 10 ** (-Q8_DECADES), torch.zeros_like(code), code.clamp(1, 255))
    deq = torch.where(code == 0, torch.zeros_like(r), 10 ** ((code - 1) * Q8_DECADES / 254 - Q8_DECADES)) * amax
    flat.copy_(deq.view(-1)[:n])
    return v


class AdamFamily(Optimizer):
    def __init__(self, params, lr=3e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.05, mode="full"):
        assert mode in ("full", "q8v") + _MATRIX_MODES
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, mode=mode)
        super().__init__(params, defaults)
        self.last_update_rms = float("nan")
        self.part = {}        # minipart only: {id(p): row -> block index (LongTensor)}; absent -> per-coordinate

    @staticmethod
    def _vmode(mode, use_matrix):
        """Geometry actually stored in st["v"] for this param (R3: explicit, never inferred from numel)."""
        if use_matrix and mode == "rowperm":
            return "rowonly"
        if mode == "minipart":
            return "minipart" if use_matrix else "full"
        return mode if (use_matrix and mode in ("scalar", "dimbal", "rowonly")) else "full"

    @torch.no_grad()
    def step(self, closure=None):
        """Two-phase step: (1) update moments and build every param's adaptive direction, (2) apply.
        R3 controls, all optional (unset -> identical to the R2 optimizer):
          group["wd_lr"]            lr used for the decoupled decay only (keeps decay fixed while the
                                    adaptive lr is rescaled for update-RMS matching)
          group["tensor_lr_mult"]   {id(p): multiplier} static per-tensor lr multipliers
          self.exact_tensor_rms     {id(p): target} same-step exact per-tensor update-RMS
          self.exact_rms_target     float, same-step exact GLOBAL adaptive update-RMS
        """
        loss = closure() if closure is not None else None
        pend = []                                         # (p, upd, lr_p, wd, wd_lr)
        for group in self.param_groups:
            b1, b2 = group["betas"]
            lr, eps, wd, mode = group["lr"], group["eps"], group["weight_decay"], group["mode"]
            wd_lr = group.get("wd_lr", lr)
            tmult = group.get("tensor_lr_mult") or {}
            factored = (mode == "adafactor")
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                g2 = g * g
                st = self.state[p]
                # compressed geometry only for true weight matrices (2D Linear) / conv (4D).
                # 3D params (e.g. mask_token (1,1,d)) have no output-channel axis -> keep per-coordinate.
                use_matrix = (p.dim() in (2, 4)) and (mode in _MATRIX_MODES)
                if mode == "minipart":                    # Adam-mini partition: only tensors with a row map
                    use_matrix = use_matrix and id(p) in self.part
                if len(st) == 0:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    if factored and use_matrix:
                        st["R"] = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)
                        st["C"] = torch.zeros(p.shape[1:].numel(), device=p.device, dtype=p.dtype)
                    elif mode == "minipart" and use_matrix:
                        st["map"] = self.part[id(p)].to(p.device)                # row -> block
                        nb = int(st["map"].max()) + 1
                        st["v"] = torch.zeros(nb, device=p.device, dtype=p.dtype)
                        st["cnt"] = torch.zeros(nb, device=p.device, dtype=p.dtype).index_add_(
                            0, st["map"], torch.ones(p.shape[0], device=p.device, dtype=p.dtype))
                    elif mode == "scalar" and use_matrix:
                        st["v"] = torch.zeros(1, device=p.device, dtype=p.dtype)
                    elif mode == "dimbal" and use_matrix:
                        st["v"] = torch.zeros(p.shape[1:].numel(), device=p.device, dtype=p.dtype)  # per-input (col)
                    elif mode in ("rowonly", "rowperm") and use_matrix:
                        st["v"] = torch.zeros(p.shape[0], device=p.device, dtype=p.dtype)            # per-output (row)
                        if mode == "rowperm":                 # fixed derangement-like shuffle, seeded by shape
                            gen = torch.Generator().manual_seed(977 + 31 * p.shape[0] + p.numel())
                            st["perm"] = torch.randperm(p.shape[0], generator=gen).to(p.device)
                    else:
                        st["v"] = torch.zeros_like(p)
                    if "v" in st:
                        st["vmode"] = self._vmode(mode, use_matrix)
                st["step"] += 1
                t = st["step"]
                m = st["m"]
                m.mul_(b1).add_(g, alpha=1 - b1)
                mhat = m / (1 - b1 ** t)
                bc = 1 - b2 ** t

                if factored and use_matrix:
                    g2f = g2.reshape(p.shape[0], -1)
                    R, C = st["R"], st["C"]
                    R.mul_(b2).add_(g2f.mean(dim=1), alpha=1 - b2)
                    C.mul_(b2).add_(g2f.mean(dim=0), alpha=1 - b2)
                    Rh, Ch = R / bc, C / bc
                    v = torch.outer(Rh, Ch) / (Rh.mean() + 1e-30)
                    denom = v.reshape(p.shape).sqrt() + eps
                else:
                    v = st["v"]
                    if mode == "minipart" and use_matrix:                 # block mean (equal row lengths)
                        red = torch.zeros_like(v).index_add_(0, st["map"], g2.reshape(p.shape[0], -1).mean(dim=1)) / st["cnt"]
                    elif mode == "scalar" and use_matrix:
                        red = g2.mean().view(1)
                    elif mode == "dimbal" and use_matrix:                 # col-only: mean over output dim
                        red = g2.reshape(p.shape[0], -1).mean(dim=0)
                    elif mode in ("rowonly", "rowperm") and use_matrix:   # row-only: mean over input dims
                        red = g2.reshape(p.shape[0], -1).mean(dim=1)
                    else:
                        red = g2
                    v.mul_(b2).add_(red, alpha=1 - b2)
                    if mode == "q8v" and v.numel() >= Q8_MIN:             # state lives in 8 bits between steps
                        quantize8_(v)
                    vhat = v / bc
                    if mode == "rowperm" and use_matrix:
                        vhat = vhat[st["perm"]]
                    if mode == "minipart" and use_matrix:
                        vhat = vhat[st["map"]]
                    if mode == "dimbal" and use_matrix:
                        denom = vhat.reshape((1,) + tuple(p.shape[1:])).sqrt() + eps
                    elif mode == "minipart" and use_matrix:
                        denom = vhat.reshape((p.shape[0],) + (1,) * (p.dim() - 1)).sqrt() + eps
                    elif mode in ("rowonly", "rowperm") and use_matrix:
                        denom = vhat.reshape((p.shape[0],) + (1,) * (p.dim() - 1)).sqrt() + eps
                    else:
                        denom = vhat.sqrt() + eps

                pend.append((p, mhat / denom, lr * tmult.get(id(p), 1.0), wd, wd_lr))

        self.last_tensor_rms, self.last_param_rms = {}, {}
        if not pend:
            return loss
        sq = torch.stack([u.pow(2).sum() for _, u, _, _, _ in pend]).double().cpu()   # one sync
        cnt = torch.tensor([u.numel() for _, u, _, _, _ in pend], dtype=torch.float64)
        lrs = torch.tensor([e[2] for e in pend], dtype=torch.float64)
        ttar = getattr(self, "exact_tensor_rms", None)
        if ttar:                                          # same-step exact per-tensor RMS
            for i, (p, _, _, _, _) in enumerate(pend):
                tg = ttar.get(id(p))
                if tg is not None and sq[i] > 0:
                    lrs[i] = tg / (sq[i] / cnt[i]).sqrt()
        gtar = getattr(self, "exact_rms_target", None)
        self.last_scale = 1.0
        if gtar:                                          # same-step exact global RMS
            cur = ((lrs.pow(2) * sq).sum() / cnt.sum()).sqrt()
            if cur > 0 and torch.isfinite(cur):
                self.last_scale = float(gtar / cur)
                lrs = lrs * self.last_scale
        a_sq = lrs.pow(2) * sq                            # adaptive part of the update, per param
        tot = []                                          # per-param sq norm incl. decay (GPU, one sync)
        lr_l = lrs.tolist(); rms_l = (a_sq / cnt).sqrt().tolist()
        for i, (p, upd, _, wd, wd_lr) in enumerate(pend):
            lr_p = lr_l[i]
            self.last_param_rms[id(p)] = rms_l[i]
            if p.dim() in (2, 4):
                self.last_tensor_rms[id(p)] = rms_l[i]
            if wd != 0:                                   # decoupled weight-decay movement
                tot.append((lr_p * upd + (wd_lr * wd) * p).pow(2).sum())
                p.mul_(1 - wd_lr * wd)
            else:
                tot.append(upd.pow(2).sum() * lr_p ** 2)
            p.add_(upd, alpha=-lr_p)
        n = float(cnt.sum())
        self.last_update_rms = float((a_sq.sum() / n).sqrt())   # adaptive-only RMS (matching target)
        self.last_total_rms = (float(torch.stack(tot).double().sum()) / n) ** 0.5   # incl. weight decay
        return loss

    @torch.no_grad()
    def switch_geometry(self, target):
        """Moment surgery: switch the second-moment geometry to `target`, PROJECTING the current v
        (reconstruct full v from the SOURCE geometry recorded in st["vmode"], reduce to target).
        m and step are preserved. X->X is an exact no-op. Reverse directions (dimbal->rowonly) lose
        the missing axis by construction and re-accumulate. Adafactor source is unsupported."""
        assert target in ("full", "rowonly", "dimbal", "scalar")
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                if "R" in st:
                    raise ValueError("surgery from adafactor source is unsupported (R/C state)")
                if "v" not in st or p.dim() not in (2, 4):
                    continue
                src = st["vmode"]
                if src == target:
                    continue
                v = st["v"]; out = p.shape[0]; in_flat = p.shape[1:].numel()
                if src == "full":
                    vf = v.reshape(out, in_flat)
                elif src == "scalar":
                    vf = v.expand(out, in_flat)
                elif src == "rowonly":
                    vf = v.view(out, 1).expand(out, in_flat)
                else:                                     # dimbal (per-input)
                    vf = v.view(1, in_flat).expand(out, in_flat)
                if target == "full":
                    st["v"] = vf.reshape(p.shape).clone()
                elif target == "rowonly":
                    st["v"] = vf.mean(dim=1).clone()
                elif target == "dimbal":
                    st["v"] = vf.mean(dim=0).clone()
                else:
                    st["v"] = vf.mean().view(1).clone()
                st["vmode"] = target
            group["mode"] = target

    @torch.no_grad()
    def per_output_channel_scale(self, p, eff_lr=False):
        """Per-output-channel scale, length = out. eff_lr=False: mean_i sqrt(v[o,i]) (R2 readout);
        eff_lr=True: mean_i 1/(sqrt(vhat[o,i])+eps), the effective learning rate actually applied.
        Differentiated for full/rowonly/adafactor; constant for dimbal/scalar."""
        st = self.state.get(p, {})
        if not st or p.dim() < 2:
            return None
        out = p.shape[0]
        eps = self.defaults["eps"]; bc = 1 - self.defaults["betas"][1] ** max(1, st.get("step", 1))
        f = (lambda x: 1.0 / ((x / bc).clamp_min(0).sqrt() + eps)) if eff_lr else (lambda x: x.clamp_min(0).sqrt())
        if "R" in st:                                   # adafactor
            if not eff_lr:
                return st["R"].clamp_min(0).sqrt()
            Rh, Ch = st["R"] / bc, st["C"] / bc
            return f(torch.outer(Rh, Ch) / (Rh.mean() + 1e-30) * bc).mean(1)
        v = st["v"]; src = st["vmode"]
        if src == "full":
            return f(v).reshape(out, -1).mean(1)
        if src == "rowonly":
            return f(v)
        # scalar or dimbal (per-input): same scale for every output channel -> constant vector
        return torch.full((out,), float(f(v).mean()), device=p.device, dtype=p.dtype)

    def optimizer_state_bytes(self):
        tot = 0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p, {})
                for k in ("m", "v", "R", "C"):
                    if k in st:
                        if k == "v" and group["mode"] == "q8v" and st[k].numel() >= Q8_MIN:
                            tot += st[k].numel() + 4 * (-(-st[k].numel() // Q8_BLOCK))   # 1 byte/coord + absmax/block
                        else:
                            tot += st[k].numel() * st[k].element_size()
        return tot


class ExternalMatched:
    """R5 wrapper giving third-party optimizers (bitsandbytes Adam8bit, GaLore, Adam-mini) the same protocol
    as AdamFamily: the inner optimizer runs with weight_decay=0, its realized step delta is measured, optionally
    rescaled so the GLOBAL update RMS equals `exact_rms_target` (same-step, exact), and decoupled weight decay
    is applied here with `wd_lr` (base schedule), exactly as AdamFamily does."""

    def __init__(self, inner, weight_decay, nosync=False):
        self.inner, self.wd = inner, weight_decay
        # bitsandbytes calls torch.cuda.synchronize() once per parameter; on a shared GPU that costs ~3 s/step.
        # Its kernels run on the default stream, so ordering is preserved without it (verified bit-identical).
        self.nosync = nosync
        self.exact_rms_target = None
        self.last_update_rms = self.last_total_rms = float("nan"); self.last_scale = 1.0
        self.last_param_rms, self.last_tensor_rms = {}, {}

    param_groups = property(lambda self: self.inner.param_groups)
    state = property(lambda self: self.inner.state)

    def zero_grad(self, set_to_none=True):
        self.inner.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def step(self):
        ps = [p for g in self.inner.param_groups for p in g["params"] if p.grad is not None]
        g0 = self.inner.param_groups[0]; wd_lr = g0.get("wd_lr", g0["lr"])
        snap = [p.detach().clone() for p in ps]
        if self.nosync:
            real = torch.cuda.synchronize; torch.cuda.synchronize = lambda *a, **k: None
            try:
                self.inner.step()
            finally:
                torch.cuda.synchronize = real
        else:
            self.inner.step()
        deltas = [p - s for p, s in zip(ps, snap)]
        sq = torch.stack([d.pow(2).sum() for d in deltas]).double().cpu()
        cnt = torch.tensor([d.numel() for d in deltas], dtype=torch.float64)
        cur = float((sq.sum() / cnt.sum()).sqrt())
        self.last_scale = 1.0
        if self.exact_rms_target and cur > 0 and cur == cur:
            self.last_scale = self.exact_rms_target / cur
        sc, dec = self.last_scale, wd_lr * self.wd
        rms = ((sq / cnt).sqrt() * sc).tolist(); tot = []
        for i, (p, s, d) in enumerate(zip(ps, snap, deltas)):
            self.last_param_rms[id(p)] = rms[i]
            if p.dim() in (2, 4):
                self.last_tensor_rms[id(p)] = rms[i]
            tot.append((sc * d - dec * s).pow(2).sum())
            p.copy_(s * (1 - dec) + sc * d)
        n = float(cnt.sum())
        self.last_update_rms = cur * sc
        self.last_total_rms = (float(torch.stack(tot).double().sum()) / n) ** 0.5


def adam_mini_partition(named, dim_of, heads_of):
    """Row -> block maps for the Adam-mini partition (Zhang et al., 2025) on this model's parameter names.
    query/key rows of attn.in_proj_weight: one block per head; value rows, attn.out_proj, MLP (fc1/fc2) and
    projector matrices: one block per output neuron; embedding and output layers (patch embedding, predictor
    embed/out): per coordinate (no entry). named: [(name, param)]; dim_of/heads_of: param -> attention dims."""
    part = {}
    for n, p in named:
        if p.dim() != 2:
            continue                                      # conv patch embedding, biases, norms, tokens: full
        if n.endswith("attn.in_proj_weight"):
            d, h = dim_of(n), heads_of(n)
            assert p.shape[0] == 3 * d and d % h == 0
            hr = torch.arange(d) // (d // h)              # head index of each row
            part[id(p)] = torch.cat([hr, h + hr, 2 * h + torch.arange(d)])
        elif n.endswith("attn.out_proj.weight") or ".fc1." in n or ".fc2." in n or n.split(".")[0] == "2":
            part[id(p)] = torch.arange(p.shape[0])
    return part
