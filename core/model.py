"""Shared point-wise MLP encoder + query-conditioned scalar implicit field.
TODO(AGENT): keep these names; this is NOT PointNet, Transformer or standard DINO.
"""
import copy
import torch
from torch import nn
from torch.nn import functional as F


class PlaneEncoder(nn.Module):
    def __init__(self, latent_dim=128):
        super().__init__()
        self.point = nn.Sequential(nn.Linear(5, 64), nn.SiLU(), nn.Linear(64, 128),
                                   nn.SiLU(), nn.Linear(128, 128), nn.SiLU())
        self.pool = nn.Sequential(nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, latent_dim))

    def forward(self, q, sdf, phase):
        # q: [N,P,3], sdf: [N,P]; phase: scalar 0(ED)/1(ES).
        ph = torch.ones_like(sdf[..., None]) * phase
        h = self.point(torch.cat((q, sdf[..., None], ph), dim=-1))
        return self.pool(torch.cat((h.mean(1), h.amax(1)), dim=-1))


class SDFDecoder(nn.Module):
    def __init__(self, latent_dim=128, width=256):
        super().__init__()
        dim = 3 + latent_dim + 1
        self.a = nn.Sequential(nn.Linear(dim, width), nn.Softplus(beta=10),
                               nn.Linear(width, width), nn.Softplus(beta=10))
        self.b = nn.Sequential(nn.Linear(width + dim, width), nn.Softplus(beta=10),
                               nn.Linear(width, width), nn.Softplus(beta=10), nn.Linear(width, 1))

    def forward(self, q, latent, phase):
        ph = q[..., :1] * 0 + phase
        x = torch.cat((q, latent, ph), dim=-1)
        return self.b(torch.cat((self.a(x), x), dim=-1)).squeeze(-1)


class DistributionHead(nn.Module):
    def __init__(self, latent_dim=128, prototypes=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(latent_dim, 256), nn.GELU(),
                                 nn.Linear(256, 64))
        self.prototypes = nn.Linear(64, prototypes, bias=False)

    def forward(self, z):
        h = F.normalize(self.mlp(z), dim=-1, eps=1e-6)
        w = F.normalize(self.prototypes.weight, dim=-1, eps=1e-6)
        return F.linear(h, w)


def interpolate_latent(q, s, z, axis):
    """q [Q,3], strictly increasing s [N], z [N,L]; axis is unit length.
    TODO(AGENT): NEVER detach q, a, or the returned interpolated latent for Eikonal.
    searchsorted selects an interval; within it the weight is differentiable.
    Out-of-stack queries raise: extrapolation is not evidence of missing anatomy.
    """
    if len(s) < 2 or not bool(torch.all(s[1:] > s[:-1])):
        raise ValueError('Need >=2 distinct, sorted observed planes')
    pos = q @ axis
    if bool(((pos.detach() < s[0] - 1e-5) | (pos.detach() > s[-1] + 1e-5)).any()):
        raise ValueError('Query outside observed axial extent')
    k = torch.searchsorted(s.contiguous(), pos.detach().contiguous(), right=True) - 1
    k = k.clamp(0, len(s) - 2)
    a = ((pos - s[k]) / (s[k + 1] - s[k])).clamp(0, 1)
    return (1 - a[:, None]) * z[k] + a[:, None] * z[k + 1]


def leave_one_out_latent(s, z):
    # At interior plane i, left weight uses distance to RIGHT, and vice versa.
    a = (s[1:-1] - s[:-2]) / (s[2:] - s[:-2])
    return (1 - a[:, None]) * z[:-2] + a[:, None] * z[2:]


class PlaneField(nn.Module):
    def __init__(self, latent_dim=128, width=256, prototypes=256):
        super().__init__()
        self.encoder = PlaneEncoder(latent_dim)
        self.decoder = SDFDecoder(latent_dim, width)
        self.head = DistributionHead(latent_dim, prototypes)

    def field(self, q, s, z, axis, phase):
        return self.decoder(q, interpolate_latent(q, s, z, axis), phase)


def make_teacher(student):
    teacher = copy.deepcopy(student).eval()
    teacher.requires_grad_(False)
    return teacher


@torch.no_grad()
def ema_update(teacher, student, momentum):
    # TODO(AGENT): called only after a SUCCESSFUL optimizer.step().
    for name, p in teacher.named_parameters():
        p.mul_(momentum).add_(dict(student.named_parameters())[name], alpha=1 - momentum)
    for name, b in teacher.named_buffers():
        b.copy_(dict(student.named_buffers())[name])
