"""Experiment IV-C(a): parametrized geodesics on the 2-sphere (paper Fig. 8).

Learns geodesic paths from a single source point to several target points under
the round metric of the unit sphere, using one Geodesic Bridge network shared
across all (source, target) pairs. The learned curves are compared against the
closed-form great-circle (Fisher-Rao) geodesics in spherical coordinates.

Run:
    python -m experiments.geodesic_sphere --n_targets 4
"""
import argparse

import numpy as np
import torch

from experiments import _REPO_ROOT  # noqa: F401
from src.models.geodesic_bridge import GeodesicBridge, sphere_metric


def sphere_geodesic_length(x0, x1):
    """Ground-truth great-circle distance between spherical-coordinate points."""
    def to_euc(x):
        theta, psi = x[:, 0], x[:, 1]
        return torch.stack(
            [torch.sin(theta) * torch.cos(psi),
             torch.sin(theta) * torch.sin(psi),
             torch.cos(theta)], dim=1)

    v, w = to_euc(x0), to_euc(x1)
    cos = torch.clamp((v * w).sum(1), -1.0, 1.0)
    return torch.arccos(cos)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n_targets", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=2000)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k_envelope", type=int, default=1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=str, default="models/sphere_geodesic.pt")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # One source point, several spread-out targets (theta in [0, pi], psi in [0, 2pi]).
    source = torch.tensor([[np.pi / 4, np.pi / 4]]).float()
    x0 = source.repeat(args.n_targets, 1)
    thetas = torch.linspace(np.pi / 6, 5 * np.pi / 6, args.n_targets)
    psis = torch.linspace(3 * np.pi / 4, 7 * np.pi / 4, args.n_targets)
    x1 = torch.stack([thetas, psis], dim=1).float()

    bridge = GeodesicBridge(
        dim=2, hidden_dim=args.hidden_dim, n_layers=3,
        activation="SELU", k=args.k_envelope, w_envelope=2.0,
    )
    opt = torch.optim.Adam(bridge.parameters(), lr=args.lr)

    for epoch in range(args.n_epochs):
        opt.zero_grad()
        loss = bridge.length(x0, x1, sphere_metric, n_times=20, squared=True)
        loss.backward()
        opt.step()
        if epoch % max(1, args.n_epochs // 10) == 0:
            print(f"[sphere-geodesic] epoch {epoch:4d}  length {loss.item():.5f}")

    # Compare learned vs. ground-truth great-circle lengths.
    with torch.no_grad():
        learned = bridge.length(x0, x1, sphere_metric, n_times=200, squared=False)
    gt = sphere_geodesic_length(x0, x1)
    print("learned mean sqrt-length:", learned.item())
    print("ground-truth great-circle lengths:", gt.tolist())

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(bridge.state_dict(), args.out)
    print("Saved sphere geodesic bridge to", args.out)


if __name__ == "__main__":
    main()
