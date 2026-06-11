# Neural FIM — experiment scripts

These scripts reproduce the experiments from *"Neural FIM: Bridging Statistical
Manifolds and Generative Modeling through Fisher Geometry"*. Run them from the
repository root, e.g. `python -m experiments.train_neural_fim --dataset tree`.

| Paper section / figure | Script | What it does |
|---|---|---|
| §IV-A, Fig. 5 | `parameter_selection.py` | FIM volume over the PHATE `(bandwidth, t)` grid on the tree dataset. |
| §IV-B, Figs. 6–7 | `train_neural_fim.py` | Trains the Neural-FIM encoder (JS-distance loss, Eq. 6) and computes the continuous FIM volume & trace for `tree`, `eb`, `ipsc`, `pbmc`. |
| §IV-C(a), Fig. 8 | `geodesic_sphere.py` | Geodesic Bridges from one source to several targets on the 2-sphere, vs. great-circle ground truth. |
| §IV-C(b), Fig. 9 | `geodesic_eb.py --task path` | Geodesic paths on EB data recovering differentiation branches. |
| §IV-D(a), Fig. 4 | `geodesic_flow_sphere.py` | Geodesic flow between two distributions on the 2-sphere (OT coupling), with/without the sphere metric. |
| §IV-D(b), Fig. 10 | `geodesic_eb.py --task flow` | Geodesic flow transporting EB day 0-3 → day 24-27 and generating unseen late-stage cells. |
| Table I | `trajectory_inference.py` | Leave-one-timepoint-out trajectory inference (Cite/Multi), 1-Wasserstein, 50/100 PCA dims. |

## Reusable building blocks (`src/models/`)

- `lit_encoder.py` / `lit_losses.py` — Neural-FIM encoder `phi` and the JS-distance loss.
- `fim_noemb.py` — FIM (Eq. 5), volume, trace, eigendecomposition.
- `geodesic_bridge.py` — the **Geodesic Bridge** `C_theta(x0, x1, t)` (Algorithm 2),
  analytic sphere/Euclidean metrics, the encoder-derived FIM metric, OT coupling,
  and `train_geodesic_flow`.
- `neural_ode_geodesic.py` — the alternative neural-ODE geodesic (Eq. 8) with a
  metric-length augmentation (used for point-to-point geodesics).

## Data

- `tree`, `sphere`, `swiss_roll` — generated on the fly (PHATE / sklearn).
- `eb` — expects `data/EB/EB_{data,phate,labels}.npy` (already in the repo).
- `ipsc` — expects `src/data/ipscData.mat`; `pbmc` — `src/data/pbmc.pickle`.
- `cite` / `multi` — see the data-format note in `trajectory_inference.py`
  (`--synthetic` runs without real data).
