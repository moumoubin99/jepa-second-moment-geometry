"""Collapse diagnostics on a batch of embeddings Z (n, D)."""
import torch


def _safe_eigvals(cov):
    d = cov.shape[0]
    ridge = 1e-6 * torch.diagonal(cov).mean().clamp_min(1e-12)
    cov = cov + ridge * torch.eye(d, device=cov.device, dtype=cov.dtype)
    # Run on CPU: cuSOLVER handle creation fails under heavy GPU contention
    # (CUSOLVER_STATUS_INTERNAL_ERROR). The cov is small (D x D), so CPU is cheap
    # and numerically identical.
    cov = cov.cpu()
    try:
        return torch.linalg.eigvalsh(cov).clamp_min(0)
    except Exception:
        return torch.linalg.svdvals(cov).clamp_min(0)


@torch.no_grad()
def collapse_stats(z, dead_thr=0.02):
    z = z.float()
    if not torch.isfinite(z).all():          # diverged run -> sentinel
        d = z.shape[1]
        return {"erank": float("nan"), "erank_norm": float("nan"),
                "mean_dim_std": float("nan"), "min_dim_std": float("nan"),
                "dead_frac": 1.0, "cov_off_ratio": float("nan"),
                "std_vec": torch.zeros(d), "diverged": True}
    std = z.std(dim=0)                       # per-dim std
    mean_dim_std = std.mean().item()
    min_dim_std = std.min().item()
    dead_frac = (std < dead_thr).float().mean().item()

    zc = z - z.mean(dim=0, keepdim=True)
    cov = (zc.T @ zc) / (z.shape[0] - 1)     # D, D
    d = cov.shape[0]
    diag = torch.diagonal(cov)
    off = cov - torch.diag(diag)
    off_ratio = (off.pow(2).sum() / (cov.pow(2).sum() + 1e-12)).item()

    # effective rank (entropy of normalized eigenvalues of covariance)
    ev = _safe_eigvals(cov)
    p = ev / (ev.sum() + 1e-12)
    nz = p[p > 1e-12]
    erank = torch.exp(-(nz * nz.log()).sum()).item()
    return {
        "erank": erank,
        "erank_norm": erank / d,
        "mean_dim_std": mean_dim_std,
        "min_dim_std": min_dim_std,
        "dead_frac": dead_frac,
        "cov_off_ratio": off_ratio,
        "std_vec": std.detach().cpu(),
    }


@torch.no_grad()
def rankme(z, eps=1e-7):
    """RankMe (Garrido et al. 2023): effective rank of singular value distribution."""
    z = z.float()
    if not torch.isfinite(z).all():
        return float("nan")
    s = torch.linalg.svdvals(z.cpu())  # CPU: avoid cuSOLVER handle failure on contended GPU
    p = s / (s.sum() + eps) + eps
    return torch.exp(-(p * p.log()).sum()).item()
