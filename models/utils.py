import torch
from typing import Callable, List, Tuple

def flatten_tensors(tensors: List[torch.Tensor]) -> torch.Tensor:
    return torch.cat([t.reshape(-1) for t in tensors])

def unflatten_to(tensor_flat: torch.Tensor, template_tensors: List[torch.Tensor]) -> List[torch.Tensor]:
    out = []
    idx = 0
    for t in template_tensors:
        n = t.numel()
        out.append(tensor_flat[idx: idx + n].view_as(t))
        idx += n
    return out

def hvp_from_loss(loss: torch.Tensor, params: List[torch.Tensor], v_flat: torch.Tensor) -> torch.Tensor:
    """
    Compute Hessian-vector product H v for the parameters `params` using autograd.
    `v_flat` must be the flattened vector with same total dims as params.
    Returns flattened Hv (detached).
    """
    grads = torch.autograd.grad(loss, params, create_graph=True)
    flat_grad = flatten_tensors([g for g in grads])
    grad_v = (flat_grad * v_flat).sum()
    hv = torch.autograd.grad(grad_v, params, retain_graph=True)
    hv_flat = flatten_tensors([h.detach() for h in hv])
    return hv_flat

def lanczos(Hv_fn: Callable[[torch.Tensor], torch.Tensor], n: int, k: int, device: torch.device) -> torch.Tensor:
    """
    Lanczos iteration to compute top-k eigenvector basis Q for symmetric operator H via Hv_fn.
    Returns Q: (n, k) tensor (columns orthonormal). Simple implementation; not numerically perfect but practical.
    Hv_fn: callable that accepts flattened vector v (n,) and returns Hv (n,).
    n: dimension
    k: desired Lanczos steps (<= n)
    """
    vs = []
    betas = []
    alphas = []
    v = torch.randn(n, device=device)
    v = v / (v.norm() + 1e-12)
    vs.append(v)
    w = None

    for j in range(k):
        w = Hv_fn(vs[j])
        if j > 0:
            w = w - betas[-1] * vs[j-1]
        alpha = (vs[j] * w).sum()
        w = w - alpha * vs[j]
        # reorthogonalize against all previous vs (stable)
        if j >= 1:
            for prev in vs:
                coeff = (prev * w).sum()
                w = w - coeff * prev
        beta = w.norm()
        alphas.append(alpha)
        if beta.item() < 1e-12:
            break
        betas.append(beta)
        v_next = w / beta
        vs.append(v_next)

    # Construct V matrix
    Vk = torch.stack(vs, dim=1)  # (n, m) where m <= k+1
    m = Vk.shape[1]
    # Build tridiagonal Tk of size m x m
    Tk = torch.zeros((m, m), device=device)
    for i in range(m):
        Tk[i, i] = alphas[i].detach()
    for i in range(len(betas)):
        Tk[i, i+1] = betas[i].detach()
        Tk[i+1, i] = betas[i].detach()

    # eigendecomposition of Tk
    eigvals, eigvecs = torch.linalg.eigh(Tk)  # ascending eigenvalues
    # take top-k eigenvectors (largest eigenvalues)
    take = min(k, eigvals.numel())
    idxs = torch.argsort(eigvals, descending=True)[:take]
    Uk = eigvecs[:, idxs]  # (m, take)
    Q = Vk @ Uk  # (n, take)
    # Orthonormalize Q via QR to be safe
    Q, _ = torch.linalg.qr(Q)
    return Q  # (n, take)

def build_projection_from_Q(Q: torch.Tensor) -> Callable[[torch.Tensor], torch.Tensor]:
    """
    Given Q (n, r) whose columns are orthonormal modes, return Πnull operator as function:
    Πnull(g) = g - Q @ (Q^T g)
    """
    def proj(g_flat: torch.Tensor) -> torch.Tensor:
        coeff = Q.T @ g_flat  # (r,)
        return g_flat - (Q @ coeff)
    return proj
