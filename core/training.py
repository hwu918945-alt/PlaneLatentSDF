import torch
from .data import require_shell
from .sampling import encoder_view, supervision_view, eikonal_queries
from .model import leave_one_out_latent
from .losses import observed_loss, eikonal_loss, seam_normal_loss


def case_objective(student, teacher, distill, case, cfg, epoch):
    require_shell(case)
    q_a, d_a = encoder_view(case)
    q_b, d_b = encoder_view(case)
    # Per-plane MLP/pooling never aggregates across planes. Hidden plane code is
    # excluded from its LOO student branch; direct supervision is a separate task.
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
    for i, (q, d, boundary) in enumerate(supervision_view(case)):
        direct = direct + observed_loss(student.decoder, q, d, boundary,
                                       z[i:i+1].expand(len(q), -1), axis, phase, cfg['tau'], calibration) / len(s)
        if 0 < i < len(s) - 1:
            loo = loo + observed_loss(student.decoder, q, d, boundary,
                                     z_loo[i-1:i].expand(len(q), -1), axis, phase, cfg['tau'], calibration) / (len(s) - 2)
    field_fn = lambda q: student.field(q, s, z, axis, phase)
    ramp = min(1., max(0., (epoch - cfg['warmup_epochs'] + 1) / cfg['ramp_epochs']))
    # Separate from the delayed DINO/Seam ramp: this path is active at step zero.
    diagnostics = {}
    eik = eikonal_loss(field_fn, eikonal_queries(case, cfg['eikonal_points']), diagnostics)
    seam = z.sum() * 0
    if ramp > 0 and len(s) > 2:
        terms = []
        for i in range(1, len(s)-1):
            eps = min(cfg['seam_epsilon_mm']/case['scale_mm'],
                      float(torch.minimum(s[i]-s[i-1], s[i+1]-s[i]))/20)
            per_surface = []
            for key in ('boundary', 'boundary_endo'):
                b = case[key][i]
                ids = torch.randperm(len(b), device=z.device)[:cfg['seam_points_per_surface']]
                per_surface.append(seam_normal_loss(field_fn, b[ids], axis, eps))
            terms.append(torch.stack(per_surface).mean())
        seam = torch.stack(terms).mean()
    # Only real hidden SAX planes have a teacher target. No unlabeled gap CE.
    dino = distill(student.head(z_loo), probability[1:-1]) if len(s) > 2 else z.sum()*0
    w = cfg['weights']
    total = w['self']*direct + w['loo']*loo + w['eikonal']*eik + ramp*(w['distill']*dino + w['seam']*seam)
    logs = {name: float(value.detach()) for name, value in
            [('self', direct), ('loo', loo), ('distill', dino), ('seam', seam), ('eikonal', eik), ('total', total)]}
    logs['teacher_entropy'] = float(-(probability * probability.clamp_min(1e-9).log()).sum(-1).mean())
    logs['latent_std'] = float(z.std(dim=0, unbiased=False).mean().detach())
    logs['ramp'] = ramp
    logs['weighted_eikonal'] = float((w['eikonal']*eik).detach())
    logs.update(diagnostics)
    for key in calibration[0]:
        logs[key] = sum(row[key] for row in calibration)/len(calibration)
    return total, logs, teacher_logits.detach()
