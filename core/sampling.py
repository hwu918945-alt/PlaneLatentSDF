"""SAX shell sampling contracts. Source IDs describe proposals, never truth labels."""
import numpy as np
import torch

SOURCE_NAMES = ('epi_zero', 'epi_gaussian', 'endo_zero', 'endo_gaussian',
                'myocardium', 'cavity', 'background')
POOL_COUNTS = (512, 1536, 512, 1536, 2048, 1024, 1024)
ENCODER_COUNTS = (128, 384, 128, 384, 512, 256, 256)
SUPERVISION_COUNTS = (64, 128, 64, 128, 128, 64, 64)
SCHEMA = 'sax_myo_shell_v1'


def stratified_indices(source, quotas, generator=None):
    """Each plane/source drawn without replacement; separate calls are independent."""
    rows = []
    for plane in source:
        parts = []
        for tag, count in enumerate(quotas):
            candidates = torch.nonzero(plane == tag).flatten()
            if len(candidates) < count:
                raise ValueError(f'Source {SOURCE_NAMES[tag]} has {len(candidates)}, needs {count}')
            choice = torch.randperm(len(candidates), device=source.device, generator=generator)[:count]
            parts.append(candidates[choice])
        index = torch.cat(parts)
        rows.append(index[torch.randperm(len(index), device=source.device, generator=generator)])
    return torch.stack(rows)


def deterministic_generator(case, seed=20261010):
    # Local generator: validation cannot alter the training RNG sequence.
    return torch.Generator(device=case['q'].device).manual_seed(seed)


def encoder_view(case, deterministic=False):
    if 'sampling_source' not in case:
        return case['q'], case['sdf2d']  # historical benchmark/cache inspection only
    gen = deterministic_generator(case) if deterministic else None
    index = stratified_indices(case['sampling_source'], ENCODER_COUNTS, gen)
    row = torch.arange(len(index), device=index.device)[:, None]
    return case['q'][row, index], case['sdf2d'][row, index]


def encode_case(model, case):
    q, d = encoder_view(case, deterministic=True)
    return model.encoder(q, d, case['phase'])


def supervision_view(case):
    index = stratified_indices(case['sampling_source'], SUPERVISION_COUNTS)
    rows = []
    for i, ids in enumerate(index):
        sources = case['sampling_source'][i, ids]
        nonzero = (sources != 0) & (sources != 2)
        # Keep the two boundary means separate from non-boundary observations.
        boundary = torch.stack([case['q'][i, ids[sources == tag]] for tag in (0, 2)])
        rows.append((case['q'][i, ids[nonzero]], case['sdf2d'][i, ids[nonzero]], boundary))
    return rows


def eikonal_queries(case, count, generator=None):
    """50% uniform + 25% epi/25% endo guided; no contour index correspondence."""
    if count < 4 or count % 4:
        raise ValueError('eikonal_points must be a positive multiple of four (>=4)')
    from .mesh import stack_basis
    device = case['s'].device
    basis = torch.as_tensor(stack_basis(case['axis'].detach().cpu().numpy()),
                            dtype=case['q'].dtype, device=device)
    s, axis = case['s'], case['axis']
    boundaries = torch.stack((case['boundary'], case['boundary_endo']), dim=1)
    uniform_count = count // 2
    lateral = boundaries.reshape(-1, 3) @ basis[:, :2]
    padding = 10.0 / case['scale_mm']
    low, high = lateral.amin(0) - padding, lateral.amax(0) + padding
    uv = low + torch.rand((uniform_count, 2), device=device, generator=generator) * (high - low)
    h = s[0] + torch.rand(count, device=device, generator=generator) * (s[-1] - s[0])
    uniform = uv @ basis[:, :2].T + h[:uniform_count, None] * axis
    # Uniform height implies P(gap) proportional to physical gap width.
    gh = h[uniform_count:]
    k = (torch.searchsorted(s, gh.contiguous(), right=True) - 1).clamp(0, len(s)-2)
    side = torch.randint(2, (len(gh),), device=device, generator=generator)
    tags = torch.arange(len(gh), device=device) % 2
    j = torch.randint(boundaries.shape[2], (len(gh),), device=device, generator=generator)
    b = boundaries[k + side, tags, j]
    sigma_mm = torch.tensor([1., 3., 6.], device=device)
    choices = torch.multinomial(torch.tensor([.4, .4, .2], device=device), len(gh),
                                replacement=True, generator=generator)
    noise = torch.randn((len(gh), 2), device=device, generator=generator) * (sigma_mm[choices, None]/case['scale_mm'])
    guided = b + (gh - b @ axis)[:, None] * axis + noise @ basis[:, :2].T
    return torch.cat((uniform, guided))


def validate_training_config(cfg):
    if cfg.get('anatomy') != 'LV_myocardium_shell':
        raise ValueError('Use the SAX myocardium shell configuration')
    if cfg.get('decoder_initialization') != 'random':
        raise ValueError('Shell training requires standard random initialization')
    for key, expected in [('encoder_points', 2048), ('supervision_points', 640),
                          ('zero_points_per_surface', 64), ('seam_points_per_surface', 16)]:
        if cfg.get(key) != expected:
            raise ValueError(f'{key} must equal the agreed quota {expected}')
    if cfg.get('eikonal_points', 0) < 4 or cfg['eikonal_points'] % 4 or cfg['weights']['eikonal'] <= 0:
        raise ValueError('Eikonal must have positive weight and a query count divisible by four')
    if cfg.get('ramp_epochs', 0) <= 0 or cfg.get('seam_epsilon_mm', 0) <= 0:
        raise ValueError('Positive ramp_epochs and seam_epsilon_mm required')
    if cfg['weights'].get('geometry', 0) != 0:
        raise ValueError('Gap geometry has been removed')
