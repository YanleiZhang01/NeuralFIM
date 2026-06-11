"""Experiment IV-B: train Neural FIM and compute volume / trace (paper Figs. 6-7).

Trains the Neural-FIM encoder phi (``LitAutoencoder``) to match Jensen-Shannon
distances between PHATE diffusion probabilities (loss of Eq. 6), then computes the
continuous FIM (Eq. 5) at every point and reports its volume sqrt(|det g|) and
trace tr(g). High volume/trace mark branching points (Fig. 7).

Datasets: ``tree`` (synthetic), ``eb`` (embryoid body scRNA-seq), ``ipsc`` (mass
cytometry), ``pbmc``. Per-dataset defaults follow the paper's notebooks; override
any of them on the command line.

Run:
    python -m experiments.train_neural_fim --dataset tree
    python -m experiments.train_neural_fim --dataset eb --max_epochs 1
"""
import argparse
import os

import numpy as np
import torch
import pytorch_lightning as pl
from pytorch_lightning import Trainer

from experiments import _REPO_ROOT
from src.data.make_dataset import train_dataloader
from src.models.lit_encoder import LitAutoencoder


# Per-dataset defaults distilled from the paper's notebooks.
DATASET_DEFAULTS = {
    "tree": dict(
        n_obs=1500, n_dim=10, batch_size=150, encoder_layer=[100, 100, 2],
        emb_dim=2, kernel_type="phate", loss_emb=False, t=20, knn=5,
        scale=0.0005, max_epochs=150, inference_obs=1600,
    ),
    "eb": dict(
        n_obs=16821, n_dim=64, batch_size=1682, encoder_layer=[128, 64, 2],
        emb_dim=2, kernel_type="phate", loss_emb=True, t=20, knn=5,
        scale=0.0001, max_epochs=50, inference_obs=16821,
    ),
    "ipsc": dict(
        n_obs=1500, n_dim=33, batch_size=50, encoder_layer=[100, 100, 50],
        emb_dim=50, kernel_type="ipsc_phate", loss_emb=False, t=1, knn=5,
        scale=0.0005, max_epochs=150, inference_obs=1500,
    ),
    "pbmc": dict(
        n_obs=1500, n_dim=10, batch_size=150, encoder_layer=[100, 100, 2],
        emb_dim=2, kernel_type="pbmc_phate", loss_emb=False, t=20, knn=5,
        scale=0.0005, max_epochs=150, inference_obs=1500,
    ),
}


def compute_fim_volume_trace(encoder, X, in_dim, device):
    """Continuous FIM per point (Eq. 5); returns (volume, trace) numpy arrays.

    I_phi(x)_{i,j} = sum_k J_phi(x)_{k,i} J_phi(x)_{k,j} phi(x)_k.
    Device-agnostic (works on CPU or GPU).
    """
    n = X.shape[0]
    volume = np.zeros(n)
    trace = np.zeros(n)
    X = X.to(device)
    for k in range(n):
        xk = X[k : k + 1].detach().requires_grad_(True)
        phi = encoder(xk).squeeze(0).detach()  # (m,)
        jac = torch.autograd.functional.jacobian(
            lambda z: encoder(z).squeeze(0), xk
        ).squeeze(1)  # (m, d)
        fim = torch.einsum("ki,kj,k->ij", jac, jac, phi)  # (d, d)
        eig = torch.linalg.eigvalsh(fim.float())
        volume[k] = torch.sqrt(torch.abs(torch.prod(eig))).item()
        trace[k] = torch.sum(eig).item()
    return volume, trace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=list(DATASET_DEFAULTS), default="tree")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no_inference", action="store_true",
                        help="skip the FIM volume/trace computation after training")
    parser.add_argument("--out_dir", type=str, default="models")
    parser.add_argument(
        "--eb_data_path", type=str,
        default=os.path.join(_REPO_ROOT, "data", "EB"),
        help="directory holding EB_data.npy / EB_phate.npy / EB_labels.npy",
    )
    args = parser.parse_args()

    cfg = dict(DATASET_DEFAULTS[args.dataset])
    if args.max_epochs is not None:
        cfg["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.scale is not None:
        cfg["scale"] = args.scale

    pl.seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    accelerator = "gpu" if device == "cuda" else "cpu"

    dl_kwargs = {}
    if args.dataset == "eb":
        dl_kwargs["download_path"] = args.eb_data_path
    train_loader = train_dataloader(
        args.dataset, cfg["n_obs"], cfg["n_dim"], cfg["emb_dim"],
        cfg["batch_size"], cfg["knn"], **dl_kwargs,
    )

    model = LitAutoencoder(
        input_dim=cfg["n_dim"],
        emb_dim=cfg["emb_dim"],
        encoder_layer=list(cfg["encoder_layer"]),
        decoder_layer=[cfg["emb_dim"], 10, 10],
        activation="ReLU",
        lr=args.lr,
        kernel_type=cfg["kernel_type"],
        loss_emb=cfg["loss_emb"],
        loss_dist=True,
        loss_rec=False,
        t=cfg["t"],
        scale=cfg["scale"],
        knn=cfg["knn"],
        logp=False,
    )

    trainer = Trainer(
        max_epochs=cfg["max_epochs"],
        accelerator=accelerator,
        devices=1,
        logger=False,
        enable_checkpointing=False,
    )
    trainer.fit(model, train_dataloaders=train_loader)

    os.makedirs(args.out_dir, exist_ok=True)
    model_path = os.path.join(args.out_dir, f"neural_fim_{args.dataset}.pt")
    torch.save(model.state_dict(), model_path)
    print("Saved model to", model_path)

    if args.no_inference:
        return

    # Inference: compute the continuous FIM volume & trace on the data.
    model.to(device).eval()
    X = train_loader.dataset.X.float().to(device)
    pred = model.encode(X).detach().cpu().numpy()
    volume, trace = compute_fim_volume_trace(model.encode, X, cfg["n_dim"], device)

    out_npz = os.path.join(args.out_dir, f"neural_fim_{args.dataset}_metrics.npz")
    np.savez(out_npz, embedding=pred, volume=volume, trace=trace)
    print("Saved FIM volume/trace and embedding to", out_npz)
    print(f"  volume: min {volume.min():.3e}  max {volume.max():.3e}")
    print(f"  trace : min {trace.min():.3e}  max {trace.max():.3e}")


if __name__ == "__main__":
    main()
