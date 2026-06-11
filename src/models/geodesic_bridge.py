"""Geodesic Bridge: simulation-free parametric geodesics under a Riemannian metric.

This implements the *Geodesic Bridge* introduced in the journal version of Neural
FIM (Zhang et al., "Neural FIM: Bridging Statistical Manifolds and Generative
Modeling through Fisher Geometry"). Instead of integrating a vector field with a
neural ODE, we directly parameterize an integral curve

    C_theta(x0, x1, t) = mu(x0, x1, t) + sigma_k(t) * g_theta(x0, x1, t)

with
    mu(x0, x1, t)    = t * x1 + (1 - t) * x0          (linear interpolation)
    sigma_k(t)       = 1 - (2t - 1) ** (2k)           (boundary-respecting envelope)
    g_theta          = neural network displacement from the linear interpolation.

The whole curve is produced by a single forward pass, so training only minimizes
the curve length under a (fixed) metric g:

    L(C) = \\int_0^1 C'(t)^T g(C(t)) C'(t) dt          (energy / squared-length form)

By construction C(0) = x0 and C(1) = x1, so the boundary conditions hold exactly
and no endpoint-matching penalty is required (cf. Eq. 8 in the paper). Lifting the
curve to a coupling (x0, x1) ~ pi gives the geodesic flow of Algorithm 2.
"""

from typing import Callable, List, Optional

import numpy as np
import torch
import torch.nn as nn

try:  # POT is used for the optimal-transport minibatch coupling (Algorithm 2).
    import ot as pot
except ImportError:  # pragma: no cover - only needed for OT coupling.
    pot = None


# --------------------------------------------------------------------------- #
# Analytic metrics (used for the toy 2-sphere experiments).
# --------------------------------------------------------------------------- #
def sphere_metric(x: torch.Tensor) -> torch.Tensor:
    """Round metric of the unit 2-sphere in spherical coords (theta, psi).

    ds^2 = dtheta^2 + sin^2(theta) dpsi^2
    """
    n = x.shape[0]
    theta = x[:, 0]
    metric = torch.zeros(n, 2, 2, device=x.device, dtype=x.dtype)
    metric[:, 0, 0] = 1.0
    metric[:, 1, 1] = torch.sin(theta) ** 2
    return metric


def euclidean_metric(x: torch.Tensor) -> torch.Tensor:
    """Identity metric (flat space)."""
    n, d = x.shape
    return torch.eye(d, device=x.device, dtype=x.dtype)[None].repeat(n, 1, 1)


# --------------------------------------------------------------------------- #
# FIM metric obtained from a trained Neural-FIM encoder.
# --------------------------------------------------------------------------- #
class EncoderFIMMetric:
    """Callable that evaluates the Fisher Information Metric of a trained encoder.

    Given an encoder phi : R^d -> R^m (the Neural-FIM network), the FIM at a point
    x follows Eq. (5) of the paper:

        I_phi(x)_{i,j} = sum_k J_phi(x)_{k,i} J_phi(x)_{k,j} phi(x)_k

    The metric is detached from the encoder parameters (we only learn the geodesic),
    matching the training scheme used for the geodesic experiments.
    """

    def __init__(self, encoder: Callable[[torch.Tensor], torch.Tensor], in_dim: int):
        self.encoder = encoder
        self.in_dim = in_dim

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch = x.shape[0]
        g = torch.zeros(batch, self.in_dim, self.in_dim, device=device, dtype=x.dtype)
        for b in range(batch):
            xb = x[b : b + 1].detach().requires_grad_(True)
            phi = self.encoder(xb).squeeze(0)  # (m,)
            jac = torch.autograd.functional.jacobian(
                lambda z: self.encoder(z).squeeze(0), xb, create_graph=False
            ).squeeze(1)  # (m, d)
            g[b] = torch.einsum("ki,kj,k->ij", jac, jac, phi.detach())
        return g.detach()


# --------------------------------------------------------------------------- #
# Geodesic Bridge network.
# --------------------------------------------------------------------------- #
class GeodesicBridge(nn.Module):
    """Parametric curve C_theta(x0, x1, t) respecting the endpoint boundaries.

    Args:
        dim: ambient dimension of the points.
        hidden_dim: width of the displacement network g_theta.
        n_layers: number of hidden layers in g_theta.
        activation: activation module name in ``torch.nn`` (e.g. "SELU").
        k: order of the envelope sigma_k(t) = 1 - (2t-1)^(2k). Larger k gives more
            weight to the displacement network (see Fig. 3 of the paper).
        w_envelope: scalar multiplier on the envelope.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 64,
        n_layers: int = 3,
        activation: str = "SELU",
        k: int = 1,
        w_envelope: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.k = k
        self.w_envelope = w_envelope

        act = getattr(nn, activation)
        # g_theta takes (x0, x1, t) -> displacement in R^dim.
        layers: List[nn.Module] = [nn.Linear(2 * dim + 1, hidden_dim), act()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden_dim, hidden_dim), act()]
        layers += [nn.Linear(hidden_dim, dim)]
        self.net = nn.Sequential(*layers)

    def envelope(self, t: torch.Tensor) -> torch.Tensor:
        """sigma_k(t) = w * (1 - (2t - 1)^(2k)); zero at t in {0, 1}."""
        return self.w_envelope * (1.0 - (2.0 * t - 1.0) ** (2 * self.k))

    def forward(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Evaluate the curve.

        Shapes:
            x0, x1: (B, dim)
            t:      (B, 1)
        Returns:
            C(t): (B, dim)
        """
        mu = t * x1 + (1.0 - t) * x0
        displacement = self.net(torch.cat([x0, x1, t], dim=-1))
        return mu + self.envelope(t) * displacement

    def velocity(self, x0: torch.Tensor, x1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """dC/dt via autograd (B, dim). Differentiable but slower than ``length``."""
        t = t.clone().requires_grad_(True)
        c = self.forward(x0, x1, t)
        dc = []
        for i in range(self.dim):
            grad = torch.autograd.grad(
                c[:, i].sum(), t, create_graph=True, retain_graph=True
            )[0]
            dc.append(grad)
        return torch.cat(dc, dim=-1)

    def _path_on_grid(self, x0, x1, ts):
        """Evaluate C at every time in ``ts`` with a single forward pass.

        Returns a tensor of shape ``(T, B, dim)``.
        """
        T, B = ts.shape[0], x0.shape[0]
        x0r = x0.unsqueeze(0).expand(T, B, self.dim).reshape(T * B, self.dim)
        x1r = x1.unsqueeze(0).expand(T, B, self.dim).reshape(T * B, self.dim)
        tr = ts.view(T, 1, 1).expand(T, B, 1).reshape(T * B, 1)
        c = self.forward(x0r, x1r, tr)
        return c.view(T, B, self.dim)

    def length(
        self,
        x0: torch.Tensor,
        x1: torch.Tensor,
        metric: Callable[[torch.Tensor], torch.Tensor],
        n_times: int = 20,
        squared: bool = True,
    ) -> torch.Tensor:
        """Discretized curve length (energy) under ``metric``.

        Vectorized: a single forward pass evaluates the whole curve, finite
        differences give the segment velocities, and the metric is evaluated at the
        segment midpoints.

        L = sum_seg dC^T g(C_mid) dC / dt          (squared=True, energy form)
        or L = sum_seg sqrt(dC^T g(C_mid) dC)      (squared=False, true length).
        """
        device = x0.device
        T, B = n_times, x0.shape[0]
        ts = torch.linspace(0, 1, T, device=device)
        dt = 1.0 / (T - 1)

        path = self._path_on_grid(x0, x1, ts)          # (T, B, dim)
        seg = path[1:] - path[:-1]                       # (T-1, B, dim)
        mid = 0.5 * (path[1:] + path[:-1])               # (T-1, B, dim)

        g = metric(mid.reshape((T - 1) * B, self.dim))   # ((T-1)*B, dim, dim)
        g = g.view(T - 1, B, self.dim, self.dim).to(seg)
        quad = torch.einsum("sbi,sbij,sbj->sb", seg, g, seg)
        quad = torch.clamp(quad, min=0.0)
        if squared:
            return (quad / dt).sum(dim=0).mean()
        return torch.sqrt(quad + 1e-12).sum(dim=0).mean()

    @torch.no_grad()
    def sample_path(
        self, x0: torch.Tensor, x1: torch.Tensor, n_times: int = 100
    ) -> torch.Tensor:
        """Return the curve sampled at ``n_times`` points: (n_times, B, dim)."""
        ts = torch.linspace(0, 1, n_times, device=x0.device)
        path = [self.forward(x0, x1, tau.repeat(x0.shape[0], 1)) for tau in ts]
        return torch.stack(path, dim=0)


# --------------------------------------------------------------------------- #
# Minibatch couplings (Algorithm 2).
# --------------------------------------------------------------------------- #
def sample_coupling(
    x0: torch.Tensor, x1: torch.Tensor, method: str = "ot"
) -> tuple:
    """Sample a paired batch (x0[i], x1[j]) from a coupling pi.

    method="ot": optimal-transport plan (POT exact EMD), as in OT-CFM.
    method="independent": random independent pairing q0 (x) q1.
    """
    batch = x0.shape[0]
    if method == "independent" or pot is None:
        j = torch.randperm(x1.shape[0])[:batch]
        return x0, x1[j]

    a, b = pot.unif(x0.shape[0]), pot.unif(x1.shape[0])
    M = torch.cdist(x0, x1) ** 2
    M = M / (M.max() + 1e-12)
    pi = pot.emd(a, b, M.detach().cpu().numpy())
    p = pi.flatten()
    p = p / p.sum()
    choices = np.random.choice(pi.shape[0] * pi.shape[1], p=p, size=batch)
    i, j = np.divmod(choices, pi.shape[1])
    return x0[i], x1[j]


def train_geodesic_flow(
    bridge: GeodesicBridge,
    q0: torch.Tensor,
    q1: torch.Tensor,
    metric: Callable[[torch.Tensor], torch.Tensor],
    n_epochs: int = 500,
    batch_size: int = 64,
    lr: float = 1e-3,
    coupling: str = "ot",
    n_times: int = 20,
    squared: bool = True,
    device: Optional[str] = None,
    verbose: bool = True,
) -> GeodesicBridge:
    """Minibatch geodesic-flow training (Algorithm 2 of the paper).

    Repeatedly samples a coupling (x0, x1) ~ pi and minimizes the curve length
    under ``metric``. Because C respects the boundaries, the resulting push-forward
    interpolates q0 -> q1 (Theorem III.2).
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    bridge = bridge.to(device)
    q0, q1 = q0.to(device), q1.to(device)
    opt = torch.optim.Adam(bridge.parameters(), lr=lr)

    for epoch in range(n_epochs):
        opt.zero_grad()
        i0 = torch.randint(0, q0.shape[0], (batch_size,), device=device)
        i1 = torch.randint(0, q1.shape[0], (batch_size,), device=device)
        x0, x1 = sample_coupling(q0[i0], q1[i1], method=coupling)
        loss = bridge.length(x0, x1, metric, n_times=n_times, squared=squared)
        loss.backward()
        opt.step()
        if verbose and epoch % max(1, n_epochs // 10) == 0:
            print(f"[geodesic-flow] epoch {epoch:4d}  length {loss.item():.5f}")
    return bridge
