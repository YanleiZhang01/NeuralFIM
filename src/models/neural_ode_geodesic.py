"""Neural-ODE geodesics regularized by a Riemannian metric.

This is the alternative geodesic computation used in the Neural FIM experiments
(Fig. 9, and the point-to-point geodesics on Swiss-roll / IPSC / EB). A velocity
field f_theta(t, x) is integrated with a neural ODE; the path length under a
metric g is accumulated as an augmented ODE state and minimized together with an
endpoint-matching term:

    argmin_theta  lambda * ||x_hat1 - x1||^2 + \\int_0^1 f^T g(gamma) f dt      (Eq. 8)

The augmentation machinery (``AugmentationModule`` / ``AugmentedVectorField``)
carries the length integral alongside the state so a single ``odeint`` produces
both the endpoint and the accumulated length.

Requires ``torchdyn`` (``NeuralODE``).
"""

from typing import Callable, List, Optional

import torch
import torch.nn as nn

try:
    from torchdyn.core import NeuralODE
except ImportError:  # pragma: no cover - torchdyn only needed here.
    NeuralODE = None


ACTIVATION_MAP = {
    "relu": nn.ReLU,
    "sigmoid": nn.Sigmoid,
    "tanh": nn.Tanh,
    "selu": nn.SELU,
    "elu": nn.ELU,
    "lrelu": nn.LeakyReLU,
    "softplus": nn.Softplus,
}


# --------------------------------------------------------------------------- #
# Velocity network.
# --------------------------------------------------------------------------- #
class SimpleDenseNet(nn.Module):
    def __init__(
        self,
        input_size: int,
        target_size: int,
        activation: str = "selu",
        batch_norm: bool = False,
        hidden_dims: Optional[List[int]] = None,
    ):
        super().__init__()
        hidden_dims = hidden_dims or [64, 64, 64]
        dims = [input_size, *hidden_dims, target_size]
        layers: List[nn.Module] = []
        for i in range(len(dims) - 2):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if batch_norm:
                layers.append(nn.BatchNorm1d(dims[i + 1]))
            layers.append(ACTIVATION_MAP[activation]())
        layers.append(nn.Linear(dims[-2], dims[-1]))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class VelocityNet(SimpleDenseNet):
    """Time-conditioned velocity field f_theta(t, x)."""

    def __init__(self, dim: int, *args, **kwargs):
        super().__init__(input_size=dim + 1, target_size=dim, *args, **kwargs)

    def forward(self, t, x):
        if t.dim() < 2:
            t = t.repeat(x.shape[0])[:, None]
        return self.model(torch.cat([t, x], dim=-1))


# --------------------------------------------------------------------------- #
# Metric regularizers: each returns a per-sample scalar length increment.
# --------------------------------------------------------------------------- #
class MetricReg(nn.Module):
    """Length increment dx^T g(x) dx for an arbitrary metric callable."""

    def __init__(self, metric: Callable[[torch.Tensor], torch.Tensor]):
        super().__init__()
        self.metric = metric

    def forward(self, t, x, dx, context) -> torch.Tensor:
        return torch.einsum("bi,bij,bj->b", dx, self.metric(x).to(dx), dx)


class Augmenter(nn.Module):
    """Prepend ``augment_dims`` zero channels carrying the accumulated length."""

    def __init__(self, augment_idx: int = 1, augment_dims: int = 1):
        super().__init__()
        self.augment_idx = augment_idx
        self.augment_dims = augment_dims

    def forward(self, x: torch.Tensor, ts: torch.Tensor):
        new_dims = list(x.shape)
        new_dims[self.augment_idx] = self.augment_dims
        x = torch.cat([torch.zeros(new_dims).to(x), x], self.augment_idx)
        return x, ts


class AugmentedVectorField(nn.Module):
    """Neural-ODE rhs that also integrates the metric length."""

    def __init__(self, net: nn.Module, regs: nn.ModuleList):
        super().__init__()
        self.net = net
        self.regs = regs

    def forward(self, t, state, augmented_input=True):
        n_aug = len(self.regs)

        class _Ctx:
            pass

        with torch.set_grad_enabled(True):
            x = state
            if augmented_input:
                x = x[:, n_aug:].requires_grad_(True)
            dx = self.net(t, x)
            if n_aug == 0:
                return dx
            augs = torch.stack([r(t, x, dx, _Ctx) for r in self.regs], dim=1)
        return torch.cat([augs, dx], 1) + (0 * state if augmented_input else 0)


class _SeqModule(nn.Sequential):
    """nn.Sequential that forwards multiple inputs (x, ts)."""

    def forward(self, *inputs):
        for module in self._modules.values():
            inputs = module(*inputs)
        return inputs


def build_geodesic_node(
    dim: int,
    metric: Callable[[torch.Tensor], torch.Tensor],
    hidden_dims: Optional[List[int]] = None,
    activation: str = "selu",
    solver: str = "rk4",
    device: str = "cpu",
):
    """Construct an augmented neural ODE that integrates a metric-regularized path.

    Returns ``(aug_node, net)`` where ``aug_node(x0, t_span)`` yields
    ``(_, trajectory)`` of shape ``(len(t_span), B, 1 + dim)``. Column 0 of the
    last dimension is the accumulated length; the rest is the position.
    """
    if NeuralODE is None:
        raise ImportError("torchdyn is required for the neural-ODE geodesic.")
    net = VelocityNet(dim, hidden_dims=hidden_dims, activation=activation, batch_norm=False)
    regs = nn.ModuleList([MetricReg(metric)])
    aug_net = AugmentedVectorField(net, regs).to(device)
    aug_node = _SeqModule(
        Augmenter(augment_idx=1, augment_dims=1),
        NeuralODE(aug_net, sensitivity="adjoint", solver=solver),
    ).to(device)
    return aug_node, net


def train_geodesic_node(
    aug_node,
    x0: torch.Tensor,
    x1: torch.Tensor,
    warmup_steps: int = 150,
    main_steps: int = 300,
    n_times: int = 20,
    lr: float = 1e-4,
    length_weight: float = 1.0,
    mse_weight: float = 10.0,
    verbose: bool = True,
):
    """Two-stage training (Eq. 8): warm-up to stay near x0, then match x1 + shorten.

    ``x0`` and ``x1`` are ``(B, dim)`` batches; multiple endpoints are supported by
    stacking start/end points along the batch (Fig. 8/9 of the paper).
    """
    opt = torch.optim.AdamW(aug_node.parameters(), lr=lr)

    for it in range(warmup_steps):
        opt.zero_grad()
        _, out = aug_node(x0, torch.linspace(0, 1, 5).to(x0))
        pred = out[-1]
        _, predx = pred[:, 0], pred[:, 1:]
        loss = 20.0 * nn.MSELoss()(predx, x0)
        loss.backward()
        opt.step()
        if verbose and it % max(1, warmup_steps // 3) == 0:
            print(f"[geodesic-node][warmup] {it:4d}  loss {loss.item():.5f}")

    history = []
    for it in range(main_steps):
        opt.zero_grad()
        _, out = aug_node(x0, torch.linspace(0, 1, n_times).to(x0))
        pred = out[-1]
        length, predx = pred[:, 0], pred[:, 1:]
        len_loss = torch.mean(length)
        mse = nn.MSELoss()(predx, x1)
        loss = length_weight * len_loss + mse_weight * mse
        loss.backward()
        opt.step()
        history.append(loss.item())
        if verbose and it % max(1, main_steps // 10) == 0:
            print(
                f"[geodesic-node][main] {it:4d}  loss {loss.item():.5f} "
                f"len {len_loss.item():.5f} mse {mse.item():.5f}"
            )
    return aug_node, history
