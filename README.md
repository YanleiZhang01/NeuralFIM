NeuralFIM
==============================

NeuralFIM is a neural-network method for computing a Riemannian metric (the Fisher
Information Metric, FIM) on point-cloud data. It learns a continuous, differentiable
map from data into a statistical manifold by matching Jensen-Shannon distances
between diffusion (PHATE) probabilities, and reads off the FIM from the network
Jacobian. From the learned metric we compute geometric quantities — volume, trace,
and geodesics — and we introduce the **Geodesic Bridge**, a simulation-free
parametric curve for geodesic inference between points and between distributions.

This repository contains the training code for all experiments in the paper
*"Neural FIM: Bridging Statistical Manifolds and Generative Modeling through Fisher
Geometry"* (Zhang et al.). The preliminary version appeared at ICML 2023
([arXiv:2306.06062](https://arxiv.org/abs/2306.06062)).

Running the experiments
-----------------------

All experiments are plain Python scripts under `experiments/`, runnable from the
repository root. See `experiments/README.md` for the full script ↔ paper mapping.

```bash
# §IV-A, Fig. 5  — PHATE parameter selection via FIM volume
python -m experiments.parameter_selection

# §IV-B, Figs. 6-7 — Neural FIM embeddings, volume & trace
python -m experiments.train_neural_fim --dataset tree     # also: eb, ipsc, pbmc

# §IV-C, Fig. 8 — geodesic bridges on the 2-sphere
python -m experiments.geodesic_sphere

# §IV-C/D, Figs. 9-10 — geodesic paths & flows on EB data
python -m experiments.geodesic_eb --task path
python -m experiments.geodesic_eb --task flow

# §IV-D, Fig. 4 — geodesic flow between distributions on the 2-sphere
python -m experiments.geodesic_flow_sphere --with_metric

# Table I — single-cell trajectory inference (Cite/Multi, leave-one-timepoint-out)
python -m experiments.trajectory_inference --dataset cite --dim 50
```

Code layout
-----------

- `src/models/lit_encoder.py`, `src/models/lit_losses.py` — Neural-FIM encoder and
  the JS-distance training loss (Eq. 6).
- `src/fim_noemb.py` — FIM (Eq. 5), volume, trace, eigendecomposition.
- `src/models/geodesic_bridge.py` — the Geodesic Bridge `C_theta(x0, x1, t)`
  (Algorithm 2), analytic and FIM-derived metrics, OT coupling, flow training.
- `src/models/neural_ode_geodesic.py` — alternative neural-ODE geodesic (Eq. 8).
- `src/data/make_dataset.py` — dataset loaders (tree, sphere, swiss roll, EB, IPSC,
  PBMC).
- `experiments/` — one script per paper experiment.

Environment: use `NeuralFIM.yml` to recreate the conda environment.
