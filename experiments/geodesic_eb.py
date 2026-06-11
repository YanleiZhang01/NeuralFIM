"""Experiments IV-C(b) & IV-D(b): geodesics and geodesic flow on EB data.

Uses a trained Neural-FIM encoder to define the FIM metric on the embryoid-body
(EB) single-cell dataset, then:

  * ``path`` (Fig. 9): learns geodesic bridges from one starting cell to several
    target cells, recovering differentiation branches.
  * ``flow`` (Fig. 10): trains a geodesic flow transporting the day 0-3 cell
    distribution to the day 24-27 distribution with an OT coupling; sampling along
    the learned curves generates new (unseen) late-stage cell states.

The encoder is loaded from ``--encoder`` if present, otherwise trained on the fly
via ``experiments.train_neural_fim``.

Run:
    python -m experiments.geodesic_eb --task flow
    python -m experiments.geodesic_eb --task path --n_targets 5
"""
import argparse
import os

import numpy as np
import torch

from experiments import _REPO_ROOT
from src.models.lit_encoder import LitAutoencoder
from src.models.geodesic_bridge import (
    GeodesicBridge,
    EncoderFIMMetric,
    train_geodesic_flow,
)

DAY_LABELS = ["Day 00-03", "Day 06-09", "Day 12-15", "Day 18-21", "Day 24-27"]
DAY_MAP = {lbl: i for i, lbl in enumerate(DAY_LABELS)}


def load_eb(data_dir):
    data = np.load(os.path.join(data_dir, "EB_data.npy"))
    labels = np.load(os.path.join(data_dir, "EB_labels.npy"), allow_pickle=True)
    y = np.array([DAY_MAP[l] for l in labels])
    return torch.tensor(data).float(), y


def get_encoder(args, n_dim, device):
    """Load a trained EB encoder, or train one quickly if none is provided."""
    model = LitAutoencoder(
        input_dim=n_dim, emb_dim=2, encoder_layer=[128, 64, 2],
        decoder_layer=[2, 10, 10], activation="ReLU", lr=1e-4,
        kernel_type="phate", loss_emb=True, loss_dist=True, loss_rec=False,
        t=20, scale=1e-4, knn=5, logp=False,
    )
    if args.encoder and os.path.exists(args.encoder):
        model.load_state_dict(torch.load(args.encoder, map_location=device))
        print("Loaded encoder from", args.encoder)
    else:
        from pytorch_lightning import Trainer
        from src.data.make_dataset import train_dataloader

        print("No encoder provided; training a Neural-FIM encoder on EB ...")
        loader = train_dataloader(
            "eb", 16821, n_dim, 2, 1682, 5, download_path=args.data_dir
        )
        trainer = Trainer(
            max_epochs=args.encoder_epochs,
            accelerator="gpu" if device == "cuda" else "cpu",
            devices=1, logger=False, enable_checkpointing=False,
        )
        trainer.fit(model, train_dataloaders=loader)
    return model.to(device).eval()


def run_path(args, X, y, metric, device):
    """Fig. 9: geodesic bridges from a day 0-3 cell to several later-stage cells."""
    rng = np.random.default_rng(args.seed)
    src_idx = rng.choice(np.where(y == 0)[0])
    tgt_pool = np.where(y >= 3)[0]
    tgt_idx = rng.choice(tgt_pool, size=args.n_targets, replace=False)

    x0 = X[src_idx][None].repeat(args.n_targets, 1).to(device)
    x1 = X[tgt_idx].to(device)

    bridge = GeodesicBridge(dim=X.shape[1], hidden_dim=128, n_layers=3,
                            activation="SELU", k=1, w_envelope=2.0).to(device)
    opt = torch.optim.Adam(bridge.parameters(), lr=args.lr)
    for epoch in range(args.n_epochs):
        opt.zero_grad()
        loss = bridge.length(x0, x1, metric, n_times=args.n_times, squared=True)
        loss.backward()
        opt.step()
        if epoch % max(1, args.n_epochs // 10) == 0:
            print(f"[eb-geodesic] epoch {epoch:4d}  length {loss.item():.5f}")

    paths = bridge.sample_path(x0, x1, n_times=50).detach().cpu().numpy()
    out = os.path.join(args.out_dir, "eb_geodesic_paths.npy")
    os.makedirs(args.out_dir, exist_ok=True)
    np.save(out, paths)
    print("Saved geodesic paths", paths.shape, "to", out)


def run_flow(args, X, y, metric, device):
    """Fig. 10: geodesic flow from the day 0-3 to the day 24-27 distribution."""
    q0 = X[y == 0].to(device)
    q1 = X[y == 4].to(device)
    print(f"source (day 0-3): {q0.shape[0]} cells; target (day 24-27): {q1.shape[0]} cells")

    bridge = GeodesicBridge(dim=X.shape[1], hidden_dim=128, n_layers=3,
                            activation="SELU", k=1, w_envelope=2.0)
    bridge = train_geodesic_flow(
        bridge, q0, q1, metric,
        n_epochs=args.n_epochs, batch_size=args.batch_size,
        coupling=args.coupling, n_times=args.n_times, device=device,
    )
    os.makedirs(args.out_dir, exist_ok=True)
    out = os.path.join(args.out_dir, "eb_geodesic_flow.pt")
    torch.save(bridge.state_dict(), out)
    print("Saved geodesic-flow bridge to", out)

    # Generate new late-stage cells by sampling source cells and pushing them along.
    idx = torch.randint(0, q0.shape[0], (args.batch_size,), device=device)
    x0 = q0[idx]
    x1 = q1[torch.randint(0, q1.shape[0], (args.batch_size,), device=device)]
    generated = bridge.sample_path(x0, x1, n_times=50)[-1].detach().cpu().numpy()
    gen_out = os.path.join(args.out_dir, "eb_generated_day24_27.npy")
    np.save(gen_out, generated)
    print("Saved generated late-stage cells", generated.shape, "to", gen_out)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["path", "flow"], default="flow")
    parser.add_argument("--data_dir", type=str,
                        default=os.path.join(_REPO_ROOT, "data", "EB"))
    parser.add_argument("--encoder", type=str, default="models/neural_fim_eb.pt")
    parser.add_argument("--encoder_epochs", type=int, default=50)
    parser.add_argument("--n_targets", type=int, default=5)
    parser.add_argument("--n_epochs", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--n_times", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--coupling", choices=["ot", "independent"], default="ot")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    X, y = load_eb(args.data_dir)
    encoder_model = get_encoder(args, X.shape[1], device)
    metric = EncoderFIMMetric(encoder_model.encode, in_dim=X.shape[1])

    if args.task == "path":
        run_path(args, X, y, metric, device)
    else:
        run_flow(args, X, y, metric, device)


if __name__ == "__main__":
    main()
