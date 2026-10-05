"""Compact I-JEPA: ViT context/target encoders + predictor + multi-block masking.

Kept deliberately small so a full pilot fits a 16 GB GPU in minutes. The design
follows I-JEPA (Assran et al., 2023): predict EMA-target latent representations of
masked target blocks from a visible context block, in representation space.
"""
import copy
import math
import torch
import torch.nn as nn


def sincos_2d_posembed(grid, dim):
    """Fixed 2D sin-cos positional embedding, shape (grid*grid, dim)."""
    assert dim % 4 == 0
    g = torch.arange(grid, dtype=torch.float32)
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    quarter = dim // 4
    omega = 1.0 / (10000 ** (torch.arange(quarter, dtype=torch.float32) / quarter))
    out = []
    for pos in (yy.reshape(-1), xx.reshape(-1)):
        a = pos[:, None] * omega[None, :]
        out += [torch.sin(a), torch.cos(a)]
    return torch.cat(out, dim=1)  # (grid*grid, dim)


class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, h)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(h, dim)

    def forward(self, x):
        n = self.norm1(x)
        a, _ = self.attn(n, n, n, need_weights=False)
        x = x + a
        m = self.fc2(self.act(self.fc1(self.norm2(x))))
        return x + m


class ViTEncoder(nn.Module):
    def __init__(self, img=32, patch=4, dim=192, depth=12, heads=3, in_ch=3):
        super().__init__()
        self.grid = img // patch
        self.n_patches = self.grid ** 2
        self.dim = dim
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)
        self.register_buffer("pos", sincos_2d_posembed(self.grid, dim))
        self.blocks = nn.ModuleList([Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

    def patchify(self, x):
        p = self.proj(x)  # B, dim, g, g
        return p.flatten(2).transpose(1, 2)  # B, N, dim

    def forward(self, x, keep_idx=None):
        """If keep_idx (B, K) given, encode only those patch tokens (context)."""
        tok = self.patchify(x) + self.pos[None]
        if keep_idx is not None:
            tok = torch.gather(tok, 1, keep_idx[..., None].expand(-1, -1, self.dim))
        for blk in self.blocks:
            tok = blk(tok)
        return self.norm(tok)


class Predictor(nn.Module):
    """Lightweight predictor mapping context tokens -> target-position latents.

    `depth=0` => shallow 2-layer MLP head shared per mask token (stress setting).
    """
    def __init__(self, dim, grid, pred_dim=96, depth=0, heads=3):
        super().__init__()
        self.dim = dim
        self.embed = nn.Linear(dim, pred_dim)
        self.register_buffer("pos", sincos_2d_posembed(grid, pred_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, pred_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.depth = depth
        if depth > 0:
            self.blocks = nn.ModuleList([Block(pred_dim, heads) for _ in range(depth)])
            self.pred_norm = nn.LayerNorm(pred_dim)
            self.out = nn.Linear(pred_dim, dim)
        else:  # MLP head
            self.mlp = nn.Sequential(
                nn.LayerNorm(pred_dim), nn.Linear(pred_dim, pred_dim * 2),
                nn.GELU(), nn.Linear(pred_dim * 2, dim))

    def forward(self, ctx, ctx_idx, tgt_idx):
        B = ctx.shape[0]
        c = self.embed(ctx) + self.pos[ctx_idx]
        m = self.mask_token + self.pos[tgt_idx]  # B, T, pred_dim
        if self.depth > 0:
            x = torch.cat([c, m], dim=1)
            for blk in self.blocks:
                x = blk(x)
            x = self.pred_norm(x)
            pred = self.out(x[:, c.shape[1]:])
        else:
            pred = self.mlp(m)
        return pred  # B, T, dim


class IJEPA(nn.Module):
    def __init__(self, img=32, patch=4, dim=192, depth=12, heads=3,
                 pred_dim=96, pred_depth=0, ema=0.95, vicreg_coef=0.0, reg_type="pooled"):
        super().__init__()
        self.encoder = ViTEncoder(img, patch, dim, depth, heads)
        self.grid = self.encoder.grid
        self.target = copy.deepcopy(self.encoder)
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(dim, self.grid, pred_dim, pred_depth, heads)
        self.ema = ema
        self.dim = dim
        self.vicreg_coef = vicreg_coef   # >0 enables VICReg anti-collapse reg (MA3 control)
        self.reg_type = reg_type         # R5: pooled | proj | sigreg
        self.projector = None
        if reg_type == "proj" and vicreg_coef > 0:   # standard VICReg expander (trained, discarded at eval)
            self.projector = nn.Sequential(nn.Linear(dim, 512), nn.BatchNorm1d(512), nn.ReLU(inplace=True),
                                           nn.Linear(512, 512))

    @torch.no_grad()
    def ema_update(self, m=None):
        m = self.ema if m is None else m
        for tp, sp in zip(self.target.parameters(), self.encoder.parameters()):
            tp.mul_(m).add_(sp, alpha=1 - m)
        for tb, sb in zip(self.target.buffers(), self.encoder.buffers()):
            tb.copy_(sb)

    def forward(self, x, ctx_idx, tgt_idx):
        # targets: full image through EMA target encoder, then select target patches
        with torch.no_grad():
            full = self.target(x)                     # B, N, dim
            tgt = torch.gather(full, 1, tgt_idx[..., None].expand(-1, -1, self.dim))
            tgt = nn.functional.layer_norm(tgt, (self.dim,))  # I-JEPA normalizes targets
        ctx = self.encoder(x, keep_idx=ctx_idx)       # B, K, dim
        pred = self.predictor(ctx, ctx_idx, tgt_idx)  # B, T, dim
        loss = nn.functional.smooth_l1_loss(pred, tgt)
        self.last_pred = float(loss.detach()); self.last_reg = None
        if self.vicreg_coef > 0:                      # MA3 control: standard anti-collapse regularizer
            with torch.autocast("cuda", enabled=False):   # R3: covariance in fp32, not bf16
                z = self.embed_tokens(ctx).float()
                if self.reg_type == "proj":
                    reg = self._vicreg(self.projector(z), cov_w=0.04)   # VICReg weights 25 (var) : 1 (cov)
                elif self.reg_type == "sigreg":
                    reg = self._sigreg(z)
                else:
                    reg = self._vicreg(z)
            self.last_reg = float(reg.detach())
            loss = loss + self.vicreg_coef * reg
        return loss

    def embed_tokens(self, ctx):
        """Mean-pool context tokens to a per-sample embedding (B, dim) for the VICReg term."""
        return ctx.mean(dim=1)

    def _vicreg(self, z, gamma=1.0, eps=1e-4, cov_w=1.0):
        """VICReg variance + covariance terms (Bardes et al. 2021) on embeddings z (B, dim).
        Variance hinge pushes per-dim std toward gamma; covariance penalizes off-diagonal
        correlations. (No invariance term: JEPA's smooth-L1 prediction already provides it.)"""
        z = z.float()
        std = torch.sqrt(z.var(dim=0) + eps)
        var_loss = torch.relu(gamma - std).mean()
        zc = z - z.mean(dim=0, keepdim=True)
        cov = (zc.T @ zc) / (z.shape[0] - 1)          # dim, dim
        d = cov.shape[0]
        cov_loss = (cov.pow(2).sum() - cov.diagonal().pow(2).sum()) / d
        return var_loss + cov_w * cov_loss

    def _sigreg(self, z, n_slices=256, n_knots=17, t_max=5.0):
        """SIGReg (Balestriero & LeCun 2025, LeJEPA): Epps-Pulley characteristic-function distance between
        random 1-D projections of z (B, dim) and N(0,1), averaged over `n_slices` fresh random directions."""
        z = z.float()
        A = torch.randn(z.shape[1], n_slices, device=z.device)
        A = A / A.norm(dim=0, keepdim=True)
        t = torch.linspace(-t_max, t_max, n_knots, device=z.device)
        phi = torch.exp(-0.5 * t * t)                               # N(0,1) characteristic function (= weight)
        xt = (z @ A).unsqueeze(-1) * t                              # B, S, K
        err = ((xt.cos().mean(0) - phi).pow(2) + xt.sin().mean(0).pow(2)) * phi
        return (torch.trapz(err, t, dim=-1) * z.shape[0]).mean()

    @torch.no_grad()
    def embed(self, x):
        """Mean-pooled context-encoder representation used for collapse diagnostics."""
        if x.shape[0] > 1024 and not torch.is_grad_enabled():   # chunk large eval sets: bounded peak memory
            return torch.cat([self.encoder(c).mean(dim=1) for c in x.split(1024)], dim=0)
        tok = self.encoder(x)             # B, N, dim
        return tok.mean(dim=1)            # B, dim


def sample_masks(batch, grid, n_target=4, tgt_scale=(0.15, 0.2),
                 ctx_scale=(0.85, 1.0), aspect=(0.75, 1.5), device="cuda", gen=None):
    """Return (ctx_idx list-of-tensors is ragged -> pad) we instead return fixed-size
    by sampling a SHARED mask for the whole batch (common I-JEPA pilot simplification).

    Returns ctx_idx (1,K) and tgt_idx (1,T) broadcastable to batch.
    """
    N = grid * grid
    def rand(a, b):
        return a + (b - a) * torch.rand(1, generator=gen, device="cpu").item()

    def block_indices(scale):
        area = max(1, int(round(rand(*scale) * N)))
        ar = math.exp(rand(math.log(aspect[0]), math.log(aspect[1])))
        h = int(round(math.sqrt(area / ar)))
        w = int(round(math.sqrt(area * ar)))
        h = min(max(h, 1), grid); w = min(max(w, 1), grid)
        top = int(rand(0, grid - h + 1)); left = int(rand(0, grid - w + 1))
        ys = torch.arange(top, top + h); xs = torch.arange(left, left + w)
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        return (yy * grid + xx).reshape(-1)

    tgt = torch.cat([block_indices(tgt_scale) for _ in range(n_target)])
    tgt = torch.unique(tgt)
    ctx = block_indices(ctx_scale)
    mask = torch.ones(N, dtype=torch.bool); mask[tgt] = False
    ctx = ctx[mask[ctx]]  # remove target patches from context
    if ctx.numel() == 0:  # fallback: keep all non-target patches
        ctx = torch.arange(N)[mask]
    return ctx.to(device)[None], tgt.to(device)[None]
