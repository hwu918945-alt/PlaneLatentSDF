"""MLPs only: no mask encoder, SDF head, or representation teacher."""
import torch
from torch import nn


class OccupancyDecoder(nn.Module):
    # Architecture copied from supplied core/model.py::SDFDecoder.
    # Five Linear, four Softplus(beta=10), same input concatenation skip.
    def __init__(self, latent_dim=128, width=256):
        super().__init__()
        dim = 3 + latent_dim + 1
        self.a = nn.Sequential(nn.Linear(dim, width), nn.Softplus(beta=10),
                               nn.Linear(width, width), nn.Softplus(beta=10))
        self.b = nn.Sequential(nn.Linear(width + dim, width), nn.Softplus(beta=10),
                               nn.Linear(width, width), nn.Softplus(beta=10), nn.Linear(width, 1))
        nn.init.normal_(self.b[-1].weight, std=0.01)
        nn.init.zeros_(self.b[-1].bias)

    def forward(self, q, latent, phase):
        x = torch.cat((q, latent, q[..., :1] * 0 + phase), dim=-1)
        return self.b(torch.cat((self.a(x), x), dim=-1)).squeeze(-1)


def interpolate(q, s, u, axis):
    if len(s) < 2 or not bool(torch.all(s[1:] > s[:-1])):
        raise ValueError('Need at least two sorted observed planes')
    pos = q @ axis
    if bool(((pos.detach() < s[0] - 1e-5) | (pos.detach() > s[-1] + 1e-5)).any()):
        raise ValueError('Query outside observed axial extent; no extrapolation')
    k = (torch.searchsorted(s.contiguous(), pos.detach().contiguous(), right=True)-1).clamp(0, len(s)-2)
    a = ((pos-s[k])/(s[k+1]-s[k])).clamp(0, 1)
    return (1-a[:, None])*u[k] + a[:, None]*u[k+1]


def loo(s, u):
    a = (s[1:-1]-s[:-2])/(s[2:]-s[:-2])
    return (1-a[:, None])*u[:-2] + a[:, None]*u[2:]


class AutoOccupancy(nn.Module):
    def __init__(self, latent_dim=128, width=256):
        super().__init__()
        self.latent_dim = latent_dim
        # No phase-specific plane embeddings or per-plane residual variables.
        self.generator = nn.Sequential(nn.Linear(latent_dim+2, 128), nn.SiLU(),
                                       nn.Linear(128, 128), nn.SiLU(), nn.Linear(128, latent_dim))
        self.decoder = OccupancyDecoder(latent_dim, width)

    def plane_latents(self, code, s, phase):
        return self.generator(torch.cat((code.expand(len(s), -1), s[:, None],
                                          torch.full_like(s[:, None], float(phase))), -1))

    def field(self, q, s, u, axis, phase):
        return self.decoder(q, interpolate(q, s, u, axis), phase)
