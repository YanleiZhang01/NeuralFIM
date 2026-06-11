"""Experiment IV-A: PHATE parameter selection with the FIM (paper Fig. 5).

We render the PHATE diffusion-potential construction as a 2-D statistical manifold
parameterized by the Gaussian-kernel bandwidth ``sigma`` and the diffusion time
``t``. For each (sigma, t) we form the diffusion potential matrix, take its
Jacobian w.r.t. (sigma, t), build the 2x2 FIM (Eq. 5) and report its volume
sqrt(|det I|). Bright regions of the resulting grid (the paper finds sigma in
[60, 90], t in [0, 4]) indicate parameter combinations that change the embedding
the most.

Run:
    python -m experiments.parameter_selection --out reports/figures/param_selection.png
"""
import argparse

import numpy as np
import torch

import phate

from experiments import _REPO_ROOT  # noqa: F401  (path setup side effect)


def torch_phate_logp(X, bandwidth, t):
    """log of the t-step diffusion operator with a Gaussian kernel (differentiable).

    A real (fractional, differentiable) matrix power is used so the operator stays
    differentiable w.r.t. the diffusion time ``t``:
    P^t = Q diag(L^t) Q^{-1}, taken on the symmetric-normalized kernel.
    """
    dists = torch.norm(X[:, None] - X, dim=2, p="fro")
    kernel = torch.exp(-(dists ** 2) / bandwidth)
    D_flat = kernel.sum(axis=0)
    D_inv_sqrt = torch.diagflat(D_flat ** -0.5)
    sym = D_inv_sqrt @ kernel @ D_inv_sqrt
    L, Q = torch.linalg.eigh(sym)
    mask = L > 1e-3
    L, Q = L[mask], Q[:, mask]
    pt = (
        torch.diagflat(D_flat ** -0.5)
        @ Q
        @ torch.diagflat(torch.pow(L, t))
        @ Q.T
        @ torch.diagflat(D_flat ** 0.5)
    )
    return torch.log(pt + 1e-12)


def fim_2x2(X, bandwidth, t):
    """2x2 FIM of the diffusion potential w.r.t. (bandwidth, t)."""
    bandwidth = torch.tensor(bandwidth, requires_grad=True, dtype=torch.float64)
    t = torch.tensor(t, requires_grad=True, dtype=torch.float64)

    def fn(b, tt):
        return torch_phate_logp(X, b, tt)

    log_p = fn(bandwidth, t)
    # Forward-mode AD: ideal here since there are only two inputs (sigma, t).
    # Reverse mode would loop/materialize over all N*N outputs (slow / OOM).
    J = torch.autograd.functional.jacobian(
        fn, (bandwidth, t), vectorize=True, strategy="forward-mode"
    )
    J = torch.stack(J)  # (2, N, N)
    fim = J.unsqueeze(1) * J.unsqueeze(0) * torch.exp(log_p)
    fim = fim.sum(axis=-1).mean(axis=-1)  # (2, 2)
    return fim


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_obs", type=int, default=150, help="subsampled points")
    parser.add_argument("--n_branch", type=int, default=5)
    parser.add_argument("--branch_length", type=int, default=300)
    parser.add_argument("--bandwidth_min", type=float, default=50.0)
    parser.add_argument("--bandwidth_max", type=float, default=150.0)
    parser.add_argument("--n_bandwidth", type=int, default=11)
    parser.add_argument("--t_min", type=float, default=1.0)
    parser.add_argument("--t_max", type=float, default=15.0)
    parser.add_argument("--n_t", type=int, default=29)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", type=str, default="reports/figures/param_selection.png")
    args = parser.parse_args()

    np.random.seed(args.seed)
    tree_data, _ = phate.tree.gen_dla(
        n_dim=10, n_branch=args.n_branch, branch_length=args.branch_length
    )
    idx = np.random.randint(tree_data.shape[0], size=args.n_obs)
    X = torch.tensor(tree_data[idx], requires_grad=True, dtype=torch.float64)

    bandwidths = np.linspace(args.bandwidth_min, args.bandwidth_max, args.n_bandwidth)
    ts = np.linspace(args.t_min, args.t_max, args.n_t)
    bv, tv = np.meshgrid(bandwidths, ts)

    volumes = []
    for b, t in zip(bv.flatten(), tv.flatten()):
        fim = fim_2x2(X, b, t)
        volumes.append(torch.sqrt(torch.abs(torch.linalg.det(fim))).item())
    volumes = np.array(volumes).reshape(bv.shape)

    print("FIM-volume grid computed:", volumes.shape)
    print("max volume at (bandwidth, t) =",
          (bv.flatten()[volumes.argmax()], tv.flatten()[volumes.argmax()]))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import os

        os.makedirs(os.path.dirname(args.out), exist_ok=True)
        plt.figure(figsize=(7, 6))
        plt.pcolormesh(bandwidths, ts, volumes, shading="auto")
        plt.xlabel("bandwidth (sigma)")
        plt.ylabel("diffusion time (t)")
        plt.title("FIM volume grid")
        plt.colorbar()
        plt.savefig(args.out, dpi=150, bbox_inches="tight")
        print("Saved figure to", args.out)
    except Exception as exc:  # pragma: no cover
        print("Skipped plotting:", exc)


if __name__ == "__main__":
    main()
