"""Experiment / Table I: single-cell trajectory inference (Cite & Multi datasets).

Leave-one-timepoint-out evaluation on the Multimodal Single-cell Integration
Challenge data (NeurIPS 2022). For each held-out timepoint ``X_t`` we train a
Geodesic Bridge with the Neural-FIM metric between the neighbouring timepoints
``X_{t-1}`` and ``X_{t+1}`` (OT coupling, Algorithm 2), sample the curve midpoint
as the prediction for ``X_t``, and report the 1-Wasserstein distance to the true
held-out cells (averaged across timepoints), for 50 and 100 PCA dimensions.

Data format
-----------
``--data_dir`` should contain, for each dataset and PCA dimension, a ``.npz`` file
named ``{dataset}_pca{dim}.npz`` with keys ``X`` (concatenated cells, ``N x dim``)
and ``t`` (integer timepoint label per cell, ``N``). Use ``--synthetic`` to run a
self-contained smoke test without real data.

Run:
    python -m experiments.trajectory_inference --dataset cite --dim 50 --data_dir data/scintegration
    python -m experiments.trajectory_inference --synthetic
"""
import argparse
import os

import numpy as np
import torch

try:
    import ot as pot
except ImportError:  # pragma: no cover
    pot = None

from experiments import _REPO_ROOT
from src.models.geodesic_bridge import (
    GeodesicBridge,
    EncoderFIMMetric,
    euclidean_metric,
)
from src.models.lit_encoder import LitAutoencoder


def wasserstein1(pred: np.ndarray, gt: np.ndarray) -> float:
    """1-Wasserstein (earth mover) distance between two point sets."""
    if pot is None:
        raise ImportError("POT (python optimal transport) is required for W1.")
    a = pot.unif(pred.shape[0])
    b = pot.unif(gt.shape[0])
    M = pot.dist(pred, gt, metric="euclidean")
    return float(pot.emd2(a, b, M))


def load_timepoints(data_dir, dataset, dim):
    path = os.path.join(data_dir, f"{dataset}_pca{dim}.npz")
    blob = np.load(path)
    X, t = blob["X"], blob["t"]
    times = sorted(np.unique(t))
    return [torch.tensor(X[t == tp]).float() for tp in times]


def make_synthetic(dim, n_times=5, n_per=200, seed=0):
    """Cells moving along a smooth curve in R^dim, one cloud per timepoint."""
    rng = np.random.default_rng(seed)
    groups = []
    for ti in range(n_times):
        center = np.zeros(dim)
        center[0] = ti
        center[1] = np.sin(ti)
        groups.append(torch.tensor(center + 0.2 * rng.standard_normal((n_per, dim))).float())
    return groups


def build_fim_metric(groups, dim, device, epochs):
    """Train a Neural-FIM encoder on the pooled cells and return its FIM metric."""
    from pytorch_lightning import Trainer
    from torch.utils.data import DataLoader, TensorDataset

    X = torch.cat(groups, dim=0)
    model = LitAutoencoder(
        input_dim=dim, emb_dim=2, encoder_layer=[128, 64, 2],
        decoder_layer=[2, 10, 10], activation="ReLU", lr=1e-4,
        kernel_type="phate", loss_emb=False, loss_dist=True, loss_rec=False,
        t=20, scale=1e-4, knn=5, logp=False,
    )
    ds = TensorDataset(X, torch.zeros(X.shape[0], 2))

    class _Wrap(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, i):
            return ds[i]

    loader = DataLoader(_Wrap(), batch_size=min(256, X.shape[0]), shuffle=True)
    Trainer(
        max_epochs=epochs, accelerator="gpu" if device == "cuda" else "cpu",
        devices=1, logger=False, enable_checkpointing=False,
    ).fit(model, train_dataloaders=loader)
    model.to(device).eval()
    return EncoderFIMMetric(model.encode, in_dim=dim)


def evaluate(groups, dim, metric, args, device):
    """Leave-one-timepoint-out W1, averaged over interior timepoints."""
    scores = []
    for t in range(1, len(groups) - 1):
        q0 = groups[t - 1].to(device)
        q1 = groups[t + 1].to(device)
        gt = groups[t].cpu().numpy()

        bridge = GeodesicBridge(dim=dim, hidden_dim=128, n_layers=3,
                                activation="SELU", k=1, w_envelope=2.0).to(device)
        opt = torch.optim.Adam(bridge.parameters(), lr=args.lr)
        for _ in range(args.n_epochs):
            opt.zero_grad()
            i0 = torch.randint(0, q0.shape[0], (args.batch_size,), device=device)
            i1 = torch.randint(0, q1.shape[0], (args.batch_size,), device=device)
            loss = bridge.length(q0[i0], q1[i1], metric, n_times=args.n_times, squared=True)
            loss.backward()
            opt.step()

        # Predict the held-out timepoint as the curve midpoint (t = 0.5).
        with torch.no_grad():
            n = min(q0.shape[0], q1.shape[0], args.eval_size)
            x0 = q0[torch.randperm(q0.shape[0])[:n]]
            x1 = q1[torch.randperm(q1.shape[0])[:n]]
            mid = bridge(x0, x1, torch.full((n, 1), 0.5, device=device))
        w1 = wasserstein1(mid.cpu().numpy(), gt)
        scores.append(w1)
        print(f"  held-out t={t}: W1 = {w1:.3f}")
    return float(np.mean(scores))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=["cite", "multi"], default="cite")
    parser.add_argument("--dim", type=int, default=50,
                        help="PCA dimension; the paper reports 50 and 100")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(_REPO_ROOT, "data", "scintegration"))
    parser.add_argument("--metric", choices=["fim", "euclidean"], default="fim")
    parser.add_argument("--encoder_epochs", type=int, default=50)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_size", type=int, default=256)
    parser.add_argument("--n_times", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--synthetic", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.synthetic:
        print("Running synthetic smoke test.")
        groups = make_synthetic(args.dim)
    else:
        groups = load_timepoints(args.data_dir, args.dataset, args.dim)
    print(f"{len(groups)} timepoints, dim={args.dim}, "
          f"cells/timepoint={[g.shape[0] for g in groups]}")

    if args.metric == "fim":
        metric = build_fim_metric(groups, args.dim, device, args.encoder_epochs)
    else:
        metric = euclidean_metric

    avg_w1 = evaluate(groups, args.dim, metric, args, device)
    print(f"\n{args.dataset} (dim {args.dim}, {args.metric} metric): "
          f"average 1-Wasserstein = {avg_w1:.3f}")


if __name__ == "__main__":
    main()
