"""Per-plane Direct/LOO weighting for the existing PlaneAutoOccupancy interface."""
import math
import torch
from torch.nn import functional as F
from .data import sample_plane
from .model import loo


def mix_plane_losses(direct_losses, loo_losses, beta=0.25, data_weight=1.5):
    """loo_losses maps an interior plane index to its BCE."""
    if len(direct_losses) < 2:
        raise ValueError("Need at least two observed planes")
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("loo_mix must be in [0, 1]")
    if not math.isfinite(data_weight) or data_weight < 0.0:
        raise ValueError("data_weight must be finite and nonnegative")
    if beta > 0.0 and len(direct_losses) < 3:
        raise ValueError("LOO needs at least three observed planes")

    plane_losses = []
    for i, direct_loss in enumerate(direct_losses):
        if beta > 0.0 and 0 < i < len(direct_losses) - 1:
            if i not in loo_losses:
                raise ValueError(f"Missing LOO loss for interior plane {i}")
            plane_loss = (1.0 - beta) * direct_loss + beta * loo_losses[i]
        else:
            plane_loss = direct_loss
        plane_losses.append(plane_loss)
    return data_weight * torch.stack(plane_losses).mean()


def objective(model, code, c, cfg, step):
    # Fixed from the first update: no warmup, gate, or dynamic schedule.
    beta = float(cfg.get("loo_mix", 0.25))
    # Preserve the old loo_weight=0 switch for Direct-only/two-plane runs.
    if float(cfg["loo_weight"]) == 0.0:
        beta = 0.0
    if not math.isfinite(beta) or not 0.0 <= beta <= 1.0:
        raise ValueError("loo_mix must be in [0, 1]")
    if beta > 0.0 and len(c["s"]) < 3:
        raise ValueError("LOO needs at least three observed planes")

    # Old baseline: direct_weight=0.5, loo_weight=1.0 -> total scale 1.5.
    # Their sum supplies the total scale; loo_mix supplies the interior mix.
    data_weight = float(cfg.get(
        "data_weight", cfg["direct_weight"] + cfg["loo_weight"]
    ))
    u = model.plane_latents(code, c["s"].float(), c["phase"])
    ul = loo(c["s"], u) if beta > 0.0 else None
    directs, leaveouts, seams = [], {}, []
    dice, fallback = [], 0

    for i in range(len(c["s"])):
        q, y, count = sample_plane(c, i, cfg["points"])
        fallback += count
        logits = model.decoder(q, u[i].expand(len(q), -1), c["phase"])
        # With 50% myocardium / 25% cavity / 25% background, this mean
        # equals the explicitly weighted mean of the three region BCEs.
        directs.append(F.binary_cross_entropy_with_logits(logits, y))
        pred = (logits.detach() >= 0).float()
        dice.append(
            (2 * (pred * y).sum() + 1e-6)
            / (pred.sum() + y.sum() + 1e-6)
        )

        if 0 < i < len(c["s"]) - 1:
            if beta > 0.0:
                # Use exactly the same q/y as Direct.
                logits_loo = model.decoder(
                    q, ul[i - 1].expand(len(q), -1), c["phase"]
                )
                leaveouts[i] = F.binary_cross_entropy_with_logits(logits_loo, y)

            # Existing optional Seam implementation; baseline weight is zero.
            if cfg["seam_weight"] > 0 and step >= cfg["seam_start"]:
                eps = min(
                    cfg["seam_eps_mm"] / c["scale_mm"],
                    0.1 * float(c["s"][i] - c["s"][i - 1]),
                    0.1 * float(c["s"][i + 1] - c["s"][i]),
                )
                b = torch.cat((c["boundary"][i], c["boundary_endo"][i])).float()
                ids = torch.randint(
                    len(b), (cfg["seam_points"],), device=b.device
                )
                b = b[ids]
                lp = model.field(
                    b + eps * c["axis"], c["s"], u, c["axis"], c["phase"]
                )
                lm = model.field(
                    b - eps * c["axis"], c["s"], u, c["axis"], c["phase"]
                )
                l0 = model.field(b, c["s"], u, c["axis"], c["phase"])
                delta = (lp - 2 * l0 + lm) / (eps * c["scale_mm"])
                seams.append(F.smooth_l1_loss(
                    delta, torch.zeros_like(delta), beta=1.0
                ))

    zero = u.sum() * 0.0
    direct = torch.stack(directs).mean()
    looloss = torch.stack(list(leaveouts.values())).mean() if leaveouts else zero
    seam = torch.stack(seams).mean() if seams else zero
    zloss = code.square().mean()  # Original regularizer and units.
    data_loss = mix_plane_losses(directs, leaveouts, beta, data_weight)
    total = (
        data_loss
        + cfg["latent_weight"] * zloss
        + cfg["seam_weight"] * seam
    )

    # direct/loo remain unweighted diagnostic means, not summands of total.
    stats = dict(
        direct=float(direct.detach()),
        loo=float(looloss.detach()),
        data_loss=float(data_loss.detach()),
        loo_mix=beta,
        seam=float(seam.detach()),
        latent=float(zloss.detach()),
        sampled_direct_dice=float(torch.stack(dice).mean()),
        adjacent_latent_distance=float(
            (u[1:] - u[:-1]).norm(dim=-1).mean().detach()
        ),
        code_norm=float(code.norm().detach()),
        pool_fallbacks=fallback,
    )
    return total, stats

