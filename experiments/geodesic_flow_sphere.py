"""Experiment IV-D(a): geodesic flow between distributions on the 2-sphere (Fig. 4).

Two distributions are sampled on the unit 2-sphere (in spherical coordinates) and
a Geodesic Bridge is trained with an optimal-transport coupling to transport one
into the other while minimizing length under the round metric
``ds^2 = dtheta^2 + sin^2(theta) dpsi^2`` (Algorithm 2). Compared with plain
flow matching (Euclidean interpolation), the metric-regularized flow follows true
geodesics on the sphere.

Run:
    python -m experiments.geodesic_flow_sphere --with_metric
    python -m experiments.geodesic_flow_sphere --no-with_metric   # OTCFM baseline
"""
import argparse

import numpy as np
import torch

from experiments import _REPO_ROOT  # noqa: F401
from src.models.geodesic_bridge import (
    GeodesicBridge,
    sphere_metric,
    euclidean_metric,
    train_geodesic_flow,
)


def generate_sphere_point_cloud(num_points, radius=1.0, seed=10):
    g = torch.Generator().manual_seed(seed)
    theta = torch.rand(num_points, generator=g) * 2 * torch.pi
    phi = torch.pi * torch.rand(num_points, generator=g)
    x = radius * torch.sin(phi) * torch.cos(theta)
    y = radius * torch.sin(phi) * torch.sin(theta)
    z = radius * torch.cos(phi)
    return torch.stack([x, y, z], dim=1)


def euc_to_polar(mat):
    mat = mat.detach().cpu().numpy()
    out = np.empty((mat.shape[0], 2))
    for i, (x, y, z) in enumerate(mat):
        out[i] = (np.arctan2(np.sqrt(x ** 2 + y ** 2), z), np.arctan2(y, x))
    return torch.tensor(out).float()


def sample_around(cloud, center_polar, k):
    """Sample the k nearest sphere points (in Euclidean space) to a center point."""
    theta, psi = center_polar
    center = torch.tensor(
        [np.sin(theta) * np.cos(psi), np.sin(theta) * np.sin(psi), np.cos(theta)]
    ).float()
    d = torch.norm(cloud - center, dim=1)
    idx = torch.topk(-d, k).indices
    return euc_to_polar(cloud[idx])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with_metric", action=argparse.BooleanOptionalAction,
                        default=True, help="regularize with the sphere FIM metric")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--hidden_dim", type=int, default=64)
    parser.add_argument("--k_envelope", type=int, default=1)
    parser.add_argument("--coupling", choices=["ot", "independent"], default="ot")
    parser.add_argument("--out", type=str, default="models/sphere_flow.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    cloud = generate_sphere_point_cloud(5000, radius=1.0)
    x0 = sample_around(cloud, (np.pi / 5, np.pi / 4), args.batch_size)
    x1 = sample_around(cloud, (2 * np.pi / 5, 3 * np.pi / 4), args.batch_size)

    metric = sphere_metric if args.with_metric else euclidean_metric
    bridge = GeodesicBridge(
        dim=2, hidden_dim=args.hidden_dim, n_layers=3,
        activation="SELU", k=args.k_envelope, w_envelope=2.0,
    )
    bridge = train_geodesic_flow(
        bridge, x0, x1, metric,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        coupling=args.coupling, device="cpu",
    )

    import os

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(bridge.state_dict(), args.out)
    print("Saved geodesic-flow bridge to", args.out)

    path = bridge.sample_path(x0, x1, n_times=50)  # (T, B, 2)
    print("Sampled flow path:", tuple(path.shape),
          "(metric)" if args.with_metric else "(no metric)")


if __name__ == "__main__":
    main()
