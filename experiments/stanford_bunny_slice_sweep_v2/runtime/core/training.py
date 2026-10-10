import torch
from .data import select_view, gap_queries
from .model import leave_one_out_latent, interpolate_latent
from .losses import observed_loss, eikonal_loss, gap_geometry_loss, seam_normal_loss


def case_objective(student, teacher, distill, case, cfg, epoch):
    q_a, d_a = select_view(case, cfg['encoder_points'])
    q_b, d_b = select_view(case, cfg['encoder_points'])
    z = student.encoder(q_a, d_a, case['phase'])
    with torch.no_grad():
        z_teacher = teacher.encoder(q_b, d_b, case['phase'])
        teacher_logits = teacher.head(z_teacher)
        probability = distill.teacher_probability(teacher_logits)
    s, axis, phase = case['s'], case['axis'], case['phase']
    direct = z.sum() * 0
    loo = z.sum() * 0
    z_loo = leave_one_out_latent(s, z)
    calibration = []
    for i in range(len(s)):
        p = torch.randint(case['q'].shape[1], (cfg['supervision_points'],), device=z.device)
        j = torch.randint(case['boundary'].shape[1], (cfg['boundary_points'],), device=z.device)
        q, d, boundary = case['q'][i, p], case['sdf2d'][i, p], case['boundary'][i, j]
        direct = direct + observed_loss(student.decoder, q, d, boundary,
                                       z[i:i+1].expand(len(q), -1), axis, phase, cfg['tau'], calibration) / len(s)
        if 0 < i < len(s) - 1:
            loo = loo + observed_loss(student.decoder, q, d, boundary,
                                     z_loo[i-1:i].expand(len(q), -1), axis, phase, cfg['tau'], calibration) / (len(s) - 2)
    # No current-plane code is used to construct its leave-one-out z'_i.
    q_gap, k, a = gap_queries(case, cfg['regularization_points'], cfg['geometry_eps'])
    field_fn = lambda q: student.field(q, s, z, axis, phase)
    ramp = min(1., max(0., (epoch - cfg['warmup_epochs'] + 1) / cfg['ramp_epochs']))
    eik = eikonal_loss(field_fn, q_gap) if ramp > 0 else z.sum() * 0
    geo = gap_geometry_loss(field_fn, q_gap, axis, cfg['geometry_eps'], cfg['geometry_band']) if ramp > 0 else z.sum() * 0
    seam = z.sum() * 0
    if ramp > 0 and len(s) > 2:
        seam_boundary = case['boundary'][1:-1, ::max(1, case['boundary'].shape[1] // 16)].reshape(-1, 3)
        seam_eps = min(cfg['geometry_eps'], float((s[1:] - s[:-1]).min()) / 10)
        seam = seam_normal_loss(field_fn, seam_boundary, axis, seam_eps)
    z_gap = interpolate_latent(q_gap, s, z, axis)
    target_gap = (1 - a[:, None]) * probability[k] + a[:, None] * probability[k + 1]
    # Two custom pairings, averaged equally: real held-out plane, unlabeled gap.
    dino = distill(student.head(z_gap), target_gap)
    if len(s) > 2:
        dino = 0.5 * (dino + distill(student.head(z_loo), probability[1:-1]))
    w = cfg['weights']
    total = w['self'] * direct + w['loo'] * loo + ramp * (
        w['distill'] * dino + w['geometry'] * geo + w['seam'] * seam + w['eikonal'] * eik)
    logs = {name: float(value.detach()) for name, value in
            [('self', direct), ('loo', loo), ('distill', dino), ('geometry', geo), ('seam', seam), ('eikonal', eik), ('total', total)]}
    logs['teacher_entropy'] = float(-(probability * probability.clamp_min(1e-9).log()).sum(-1).mean())
    logs['latent_std'] = float(z.std(dim=0, unbiased=False).mean().detach())
    logs['ramp'] = ramp
    logs['calibration_valid_fraction'] = sum(calibration)/len(calibration)
    return total, logs, teacher_logits.detach()
