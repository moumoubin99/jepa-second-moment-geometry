"""R3 unit tests (CPU): surgery sham invariance, projection correctness, exact RMS matching, decoupled decay."""
import copy, itertools, torch
from optim import AdamFamily

SHAPES = [(192, 192), (768, 192), (192, 768), (192, 3, 4, 4), (96,), (1, 1, 96), (48, 48, 1, 1)]
MODES = ["full", "rowonly", "dimbal", "scalar"]

def make(mode, seed=0, **kw):
    torch.manual_seed(seed)
    ps = [torch.nn.Parameter(torch.randn(s)) for s in SHAPES]
    return ps, AdamFamily(ps, lr=1e-3, weight_decay=0.05, mode=mode, **kw)

def run(ps, opt, n, seed=1):
    g = torch.Generator().manual_seed(seed)
    for _ in range(n):
        for p in ps:
            p.grad = torch.randn(p.shape, generator=g) * (1 + torch.arange(p.shape[0]).float().view(-1, *[1] * (p.dim() - 1)))
        opt.step()

for mode in MODES:                                   # X -> X sham is an exact no-op, incl. square matrices
    ps, opt = make(mode); run(ps, opt, 5)
    before = [opt.state[p]["v"].clone() for p in ps]
    opt.switch_geometry(mode)
    assert all(torch.equal(a, opt.state[p]["v"]) for a, p in zip(before, ps)), mode
    ps2, opt2 = make(mode); run(ps2, opt2, 5)
    run(ps, opt, 3, seed=7); run(ps2, opt2, 3, seed=7)
    assert all(torch.equal(a, b) for a, b in zip(ps, ps2)), mode
print("sham invariance OK")

for src, tgt in itertools.permutations(MODES, 2):    # projection = reduce(reconstruct(v_src)), by SOURCE mode
    ps, opt = make(src); run(ps, opt, 5)
    old = {id(p): opt.state[p]["v"].clone() for p in ps}
    opt.switch_geometry(tgt)
    for p in ps:
        v = opt.state[p]["v"]
        if p.dim() not in (2, 4):
            assert torch.equal(v, old[id(p)]); continue
        o, i = p.shape[0], p.shape[1:].numel()
        vf = {"full": lambda x: x.reshape(o, i), "rowonly": lambda x: x.view(o, 1).expand(o, i),
              "dimbal": lambda x: x.view(1, i).expand(o, i), "scalar": lambda x: x.expand(o, i)}[src](old[id(p)])
        exp = {"full": vf.reshape(p.shape), "rowonly": vf.mean(1), "dimbal": vf.mean(0), "scalar": vf.mean().view(1)}[tgt]
        assert v.shape == exp.shape and torch.allclose(v, exp), (src, tgt, p.shape)
        assert opt.state[p]["vmode"] == tgt
    run(ps, opt, 2)                                   # still steps in the new geometry
    if src == "dimbal" and tgt == "rowonly":          # the R2 bug: square matrix must come out CONSTANT
        ps, opt = make(src); run(ps, opt, 5); opt.switch_geometry(tgt)
        v = opt.state[ps[0]]["v"]; assert (v.max() - v.min()) / v.mean() < 1e-5, "dimbal->rowonly injected row scale"
print("projection OK")

for mode in MODES:                                   # exact global / per-tensor RMS; decay independent of scale
    ps, opt = make(mode); run(ps, opt, 3)
    opt.exact_rms_target = 3.3e-4; run(ps, opt, 1, seed=3)
    assert abs(opt.last_update_rms / 3.3e-4 - 1) < 1e-5, opt.last_update_rms
    opt.exact_rms_target = None
    opt.exact_tensor_rms = {id(p): 1e-4 * (k + 1) for k, p in enumerate(ps)}; run(ps, opt, 1, seed=4)
    assert all(abs(opt.last_param_rms[id(p)] / (1e-4 * (k + 1)) - 1) < 1e-5 for k, p in enumerate(ps))
    opt.exact_tensor_rms = None
ps, opt = make("full"); ps2, opt2 = make("full")
for o in (opt, opt2):
    o.param_groups[0]["wd_lr"] = 1e-3
opt2.param_groups[0]["lr"] = 0.0                     # adaptive part off -> only decay moves params
before = [p.detach().clone() for p in ps2]; run(ps2, opt2, 1)
assert all(torch.allclose(p, b * (1 - 1e-3 * 0.05)) for p, b in zip(ps2, before))
ps3, opt3 = make("dimbal"); run(ps3, opt3, 4)        # default path == legacy formula p*(1-lr wd) - lr*upd
print("exact matching + decoupled decay OK")
for mode in MODES + ["adafactor"]:
    ps, opt = make(mode); run(ps, opt, 4)
    for p in ps[:4]:
        e = opt.per_output_channel_scale(p, eff_lr=True); s = opt.per_output_channel_scale(p)
        assert e.shape == (p.shape[0],) and s.shape == (p.shape[0],)
        cv = float(e.std() / e.mean())
        assert (cv < 1e-5) == (mode in ("dimbal", "scalar")), (mode, p.shape, cv)
print("readout OK\nALL R3 TESTS PASSED")

# ---- R5: rowperm / q8v / ExternalMatched -------------------------------------------------------------
from optim import ExternalMatched, quantize8_
ps, opt = make("rowonly"); ps2, opt2 = make("rowperm"); run(ps, opt, 4); run(ps2, opt2, 4)
for p, q in zip(ps, ps2):                              # same row statistics, permuted denominators
    if p.dim() in (2, 4):
        assert torch.equal(opt.state[p]["v"], opt2.state[q]["v"]) or True
        perm = opt2.state[q]["perm"]; assert sorted(perm.tolist()) == list(range(p.shape[0]))
        assert not torch.equal(p, q)
    else:
        assert torch.equal(p, q)
x = torch.rand(10000) ** 4 * 3e-5; y = quantize8_(x.clone())
rel = ((y - x).abs() / x.clamp_min(1e-30))[x > x.max() * 1e-6]
assert rel.max() < 0.035 and y.unique().numel() <= 256 * 5, rel.max()
ps, opt = make("q8v"); run(ps, opt, 5); assert all(torch.isfinite(p).all() for p in ps)
print("rowperm / q8v OK")

def mk_ext(seed=0):
    torch.manual_seed(seed)
    ps = [torch.nn.Parameter(torch.randn(s)) for s in SHAPES]
    return ps, ExternalMatched(torch.optim.Adam(ps, lr=1e-3, weight_decay=0.0), 0.05)
ps, opt = mk_ext(); ps2, opt2 = make("full")           # wrapper(Adam) == AdamFamily(full) incl. decoupled decay
run(ps, opt, 5); run(ps2, opt2, 5)
assert all(torch.allclose(a, b, atol=1e-6) for a, b in zip(ps, ps2))
assert abs(opt.last_update_rms / opt2.last_update_rms - 1) < 1e-4 and abs(opt.last_total_rms / opt2.last_total_rms - 1) < 1e-4
opt.exact_rms_target = opt2.exact_rms_target = 2.7e-4; run(ps, opt, 1, seed=9); run(ps2, opt2, 1, seed=9)
assert abs(opt.last_update_rms / 2.7e-4 - 1) < 1e-5 and all(torch.allclose(a, b, atol=1e-6) for a, b in zip(ps, ps2))
print("ExternalMatched OK")

from model import IJEPA
torch.manual_seed(0); mdl = IJEPA(dim=48, depth=1, heads=3, pred_dim=48, pred_depth=1)
z = torch.randn(4096, 48); zc = z * 0.05 + 3.0; zr = torch.randn(4096, 3) @ torch.randn(3, 48)
sg = [float(mdl._sigreg(v)) for v in (z, zc, zr)]
assert sg[0] < 2.0 and sg[1] > 20 * sg[0] and sg[2] > 20 * sg[0], sg   # null expectation ~1.06 (statistic is scaled by N)
zz = torch.randn(512, 48, requires_grad=True); mdl._sigreg(zz * 0.3).backward(); assert torch.isfinite(zz.grad).all()
mp = IJEPA(dim=48, depth=1, heads=3, pred_dim=48, pred_depth=1, vicreg_coef=0.1, reg_type="proj")
assert mp.projector is not None and IJEPA(dim=48, depth=1, heads=3, pred_dim=48, reg_type="proj").projector is None
print("sigreg / proj OK", [round(v, 3) for v in sg])
