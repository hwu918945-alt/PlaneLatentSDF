"""Differentiable losses. Float32 minimum; no autocast for coordinate gradients."""
import torch
from torch import nn
from torch.nn import functional as F


def coordinate_gradient(value, q, create_graph=True):
    return torch.autograd.grad(value.sum(), q, create_graph=create_graph, retain_graph=True)[0]


def observed_loss(decoder, q, d2, boundary, z, axis, phase, tau=0.02, diagnostics=None):
    """Same loss for direct z_i and leave-one-out z'_i.
    2D SDF is a plane distance, not an exact 3D SDF label.
    Local first-order compatibility: F ~= ||P_plane grad F|| * d2.
    TODO(AGENT): do NOT replace this with global abs(F - d2).
    """
    qr = q.detach().clone().requires_grad_(True)
    pred = decoder(qr, z, phase)
    grad = coordinate_gradient(pred, qr)
    tangent = grad - (grad * axis).sum(-1, keepdim=True) * axis
    factor = tangent.norm(dim=-1).detach()  # A fixed scale for this step, not another head.
    near = (d2.abs() > 1e-5) & (d2.abs() < tau) & (factor > 0.15)
    if diagnostics is not None:
        diagnostics.append(float(near.float().mean().detach()))
    residual = (pred - factor * d2) / tau
    distance = F.smooth_l1_loss(residual[near], torch.zeros_like(residual[near]), beta=0.2) if bool(near.any()) else pred.sum() * 0
    signmask = d2.abs() > 0.002
    signed = torch.sign(d2) * pred
    # Margin <= plane distance; do not make near-boundary sign labels impossible.
    margin = d2.abs().clamp(max=0.005)
    sign = (F.relu(margin[signmask] - signed[signmask]) / tau).mean() if bool(signmask.any()) else pred.sum() * 0
    surf = decoder(boundary, z[:1].expand(len(boundary), -1), phase).abs().mean() / tau
    return surf + 0.2 * sign + 0.2 * distance


def eikonal_loss(field_fn, q):
    qr = q.detach().clone().requires_grad_(True)
    g = coordinate_gradient(field_fn(qr), qr)
    # TODO(AGENT): field_fn must interpolate z using THIS qr, not precomputed z_m.
    return ((g.norm(dim=-1) - 1) ** 2).mean()


def normal_at(field_fn, q):
    qr = q.detach().clone().requires_grad_(True)
    value = field_fn(qr)
    g = coordinate_gradient(value, qr)
    return F.normalize(g, dim=-1, eps=1e-6), value


def gap_geometry_loss(field_fn, q, axis, eps=0.003, band=0.03):
    """Weak second-order surface normal variation in the axial tangent direction.
    t=(I-NN^T)axis; avoid dividing by ||t|| at caps. No contour correspondence.
    q must have >=eps clearance to both slab endpoints (sampler enforces it).
    """
    normal, value = normal_at(field_fn, q)
    tangent = (axis - (normal * axis).sum(-1, keepdim=True) * normal).detach()
    plus, _ = normal_at(field_fn, q + eps * tangent)
    minus, _ = normal_at(field_fn, q - eps * tangent)
    change = (plus - minus) / (2 * eps)
    weight = torch.exp(-(value.detach() / band) ** 2)
    # Each component uses Huber; large curvature is discouraged, not prohibited.
    robust = F.smooth_l1_loss(change, torch.zeros_like(change), beta=1, reduction='none').mean(-1)
    return (weight * robust).mean()


def seam_normal_loss(field_fn, boundary, axis, eps):
    left, _ = normal_at(field_fn, boundary - eps * axis)
    right, _ = normal_at(field_fn, boundary + eps * axis)
    # One-sided normals catch derivative jumps at piecewise-linear latent knots.
    return (1 - (left * right).sum(-1).clamp(-1, 1)).mean()


class DistanceWeightedDistillation(nn.Module):
    """DINO-style centering/sharpening/EMA CE, with custom geometric pairing.
    This is a research adaptation, NOT the complete standard DINO algorithm.
    """
    def __init__(self, prototypes=256, student_temp=0.1, teacher_temp=0.07, center_momentum=0.9):
        super().__init__()
        self.register_buffer('center', torch.zeros(1, prototypes))
        self.student_temp = student_temp
        self.teacher_temp = teacher_temp
        self.center_momentum = center_momentum

    def teacher_probability(self, logits):
        return F.softmax((logits.detach() - self.center) / self.teacher_temp, dim=-1).detach()

    def forward(self, student_logits, target_probability):
        return -(target_probability.detach() * F.log_softmax(student_logits / self.student_temp, dim=-1)).sum(-1).mean()

    @torch.no_grad()
    def update_center(self, logits):
        # TODO(AGENT): if adding DDP, all-reduce sum AND count across all ranks.
        batch_center = logits.detach().mean(0, keepdim=True)
        self.center.mul_(self.center_momentum).add_(batch_center, alpha=1 - self.center_momentum)
