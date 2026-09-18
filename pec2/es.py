"""Population ES on torch + layout-consistent theta packing."""
from __future__ import annotations
import numpy as np
import torch
from scipy import stats as sps

DEV = "cuda" if torch.cuda.is_available() else "cpu"


def rank_norm(f):
    f = np.asarray(f, float)
    r = sps.rankdata(f) - 1
    return r / max(1, len(f) - 1) - 0.5


def _layout(sizes):
    """Single source of truth: list of (in, out) per layer."""
    return [(sizes[i], sizes[i + 1]) for i in range(len(sizes) - 1)]


def pack(theta, sizes, device="cpu"):
    """Flat theta [D] -> (W list [in,out], b list [out])."""
    W, b, k = [], [], 0
    for (i, o) in _layout(sizes):
        n = i * o
        W.append(theta[k:k + n].reshape(i, o).to(device)); k += n
    for (i, o) in _layout(sizes):
        b.append(theta[k:k + o].reshape(1, o).to(device)); k += o
    return W, b


def unpack(W, b):
    """(W list, b list) -> flat theta [D]."""
    return torch.cat([w.reshape(-1) for w in W] + [bb.reshape(-1) for bb in b])


def flat_size(sizes):
    return sum(i * o + o for (i, o) in _layout(sizes))


def run_mlp(x, W, b):
    """x [.., in] -> [.., out] with tanh hidden, tanh output."""
    h = x
    for j in range(len(W) - 1):
        h = torch.tanh(h @ W[j] + b[j])
    return torch.tanh(h @ W[-1] + b[-1])


class ESNet:
    """Population-stacked MLP: params [P, in, out]; forward [P, N, in] -> [P, N, out]."""
    def __init__(self, sizes, P, device=DEV, seed=0):
        self.sizes, self.P, self.device = sizes, P, device
        g = torch.Generator(device="cpu").manual_seed(seed)
        self.shapes = _layout(sizes)
        self.W = [(torch.randn(P, i, o, generator=g) * 0.6).to(device)
                  for (i, o) in self.shapes]
        self.b = [torch.zeros(P, 1, o, device=device) for (i, o) in self.shapes]
        self.D = flat_size(sizes)

    def set_theta(self, theta):
        """theta: [P, D] -> W, b. Same order as pack()."""
        k = 0
        for si, (i, o) in enumerate(self.shapes):
            n = i * o
            self.W[si] = theta[:, k:k + n].reshape(self.P, i, o); k += n
        for si, (i, o) in enumerate(self.shapes):
            self.b[si] = theta[:, k:k + o].reshape(self.P, 1, o); k += o

    def forward(self, x):
        h = x
        for j in range(len(self.W) - 1):
            h = torch.tanh(torch.einsum("pni,pio->pno", h, self.W[j])
                           + self.b[j])
        return torch.tanh(torch.einsum("pni,pio->pno", h, self.W[-1]) + self.b[-1])

    @staticmethod
    def from_theta(theta, sizes, device=DEV):
        """theta [D] -> eval fn x [.., in] -> [.., out] (same layout)."""
        W, b = pack(theta, sizes, device=device)
        return lambda x: run_mlp(x, W, b)


def es_optimize(net, theta0, fitness_fn, gens, pop, sigma, lr, seed=0, log=None):
    """OpenAI-ES with antithetic sampling + rank-normalized fitness.
    fitness_fn(theta[P, D]) -> fitness[P] (higher = better)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    theta = theta0.clone()
    half = pop // 2
    hist = []
    for gen in range(gens):
        eps = torch.randn((half, net.D), generator=g).to(net.device)
        cand = torch.cat([theta + sigma * eps, theta - sigma * eps], 0)[:pop]
        fit = fitness_fn(cand)
        assert np.asarray(fit).shape == (cand.shape[0],), \
            f"fitness returned {np.asarray(fit).shape}, expected ({cand.shape[0]},)"
        f = torch.as_tensor(rank_norm(fit), device=net.device, dtype=torch.float32)
        grad = ((f[:half] - f[half:])[:, None] * eps).sum(0) / (half * sigma)
        theta = theta + lr * grad
        hist.append(float(np.mean(fit)))
        if log and gen % 10 == 0:
            log(f"    gen {gen:3d} mean {np.mean(fit):+.3f} best {np.max(fit):+.3f}")
    return theta, hist
