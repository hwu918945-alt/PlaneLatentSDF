"""NPZ integration contract; deliberately independent of unknown cache internals."""
from pathlib import Path
import numpy as np
import torch


def polygon_sdf(points, polygon):
    """Exact segment distance + ray crossing sign for ONE simple filled polygon.
    Negative inside. Points exactly on a segment have zero distance.
    Compose two separately validated polygons with max(d_epi, -d_endo) for shell.
    """
    p = np.asarray(points, dtype=np.float64)
    if len(p) > 2048:
        return np.concatenate([polygon_sdf(p[i:i+2048], polygon) for i in range(0, len(p), 2048)])
    a = np.asarray(polygon, dtype=np.float64)
    b = np.roll(a, -1, axis=0)
    v = b - a
    length2 = np.sum(v * v, axis=-1)
    if np.any(length2 < 1e-14):
        raise ValueError('Remove consecutive duplicate contour vertices')
    t = np.clip(np.sum((p[:, None] - a) * v, -1) / length2, 0, 1)
    closest = a + t[..., None] * v
    d = np.sqrt(np.min(np.sum((p[:, None] - closest) ** 2, -1), axis=1))
    py = p[:, 1:2]
    crossing = (a[None, :, 1] > py) != (b[None, :, 1] > py)
    denominator = b[:, 1] - a[:, 1]
    denominator = np.where(np.abs(denominator) < 1e-14, 1, denominator)
    intersection_x = a[None, :, 0] + (py - a[None, :, 1]) * v[None, :, 0] / denominator
    inside = np.sum(crossing & (p[:, 0:1] < intersection_x), axis=1) % 2 == 1
    return np.where(inside, -d, d).astype(np.float32)


def resample_contour(polygon, count=256):
    """Ordered arclength samples; first sample stays at input start vertex."""
    a = np.asarray(polygon, dtype=np.float64)
    v = np.roll(a, -1, axis=0) - a
    length = np.linalg.norm(v, axis=-1)
    if np.any(length < 1e-10):
        raise ValueError('Duplicate vertices')
    cumulative = np.r_[0., np.cumsum(length)]
    t = np.arange(count) / count * cumulative[-1]
    j = np.searchsorted(cumulative, t, side='right') - 1
    fraction = (t - cumulative[j]) / length[j]
    return (a[j] + fraction[:, None] * v[j]).astype(np.float32)


def load_case(path, device='cpu'):
    with np.load(path, allow_pickle=False) as f:
        required = ['q', 'sdf2d', 'boundary', 's', 'axis', 'center_mm', 'scale_mm', 'phase', 'case_id']
        if any(k not in f for k in required):
            raise ValueError(f'Missing keys in {path}: {required}')
        raw = {k: np.array(f[k]) for k in required}
        for key in ['schema', 'anatomy', 'sampling_source', 'boundary_endo', 'source_slice_indices']:
            if key in f:
                raw[key] = np.array(f[key])
    q, d, b, s, axis = (raw[k] for k in ['q', 'sdf2d', 'boundary', 's', 'axis'])
    if q.ndim != 3 or q.shape[-1] != 3 or d.shape != q.shape[:2]:
        raise ValueError('q must be [N,P,3], sdf2d [N,P]')
    if b.ndim != 3 or b.shape[0] != len(s) or b.shape[-1] != 3 or len(s) != len(q):
        raise ValueError('boundary must be [N,B,3]')
    if len(s) < 2 or np.any(np.diff(s) <= 1e-5):
        raise ValueError('Need >=2 strictly ordered nonduplicate planes')
    if axis.shape != (3,) or abs(np.linalg.norm(axis) - 1) > 1e-4:
        raise ValueError('axis must be unit vector')
    for arr in [q, d, b, s, axis, raw['center_mm'], raw['scale_mm']]:
        if not np.isfinite(arr).all():
            raise ValueError('Nonfinite cache')
    if np.max(np.abs(q @ axis - s[:, None])) > 1e-4 or np.max(np.abs(b @ axis - s[:, None])) > 1e-4:
        raise ValueError('Nonparallel planes: common-stack approximation not satisfied')
    if float(raw['scale_mm']) <= 0 or raw['center_mm'].shape != (3,):
        raise ValueError('Invalid normalization metadata')
    if int(raw['phase']) not in [0, 1]:
        raise ValueError('phase must be ED=0 or ES=1')
    result = {k: torch.as_tensor(raw[k], dtype=torch.float32, device=device) for k in ['q', 'sdf2d', 'boundary', 's', 'axis']}
    result.update(phase=int(raw['phase']), case_id=str(raw['case_id']),
                  center_mm=raw['center_mm'], scale_mm=float(raw['scale_mm']), path=str(path))
    result['anatomy'] = str(raw.get('anatomy', 'legacy_filled_region'))
    if 'schema' in raw:
        from .sampling import SCHEMA, POOL_COUNTS
        if str(raw['schema']) != SCHEMA:
            raise ValueError('Unknown shell cache schema')
        if result['anatomy'] != 'LV_myocardium_shell' or result['scale_mm'] != 100.:
            raise ValueError('Invalid shell anatomy or physical scale')
        source, endo = raw['sampling_source'], raw['boundary_endo']
        if len(s) < 3 or source.shape != d.shape or q.shape[1] != sum(POOL_COUNTS) or endo.shape != b.shape or b.shape[1] != 512:
            raise ValueError('Invalid shell pool/boundary shapes')
        if not np.isfinite(endo).all() or np.max(np.abs(endo @ axis - s[:, None])) > 1e-4:
            raise ValueError('Invalid endo geometry')
        for i in range(len(s)):
            if not np.array_equal(np.bincount(source[i], minlength=7), POOL_COUNTS):
                raise ValueError('Shell source quotas mismatch')
            zero = (source[i] == 0) | (source[i] == 2)
            if np.any(d[i, zero] != 0):
                raise ValueError('Boundary SDF must be exactly zero')
            for tag, sign in [(4, -1), (5, 1), (6, 1)]:
                if not np.all(sign * d[i, source[i] == tag] > 0):
                    raise ValueError('Shell region sign mismatch')
        result['schema'] = str(raw['schema'])
        result['sampling_source'] = torch.as_tensor(source, dtype=torch.long, device=device)
        result['boundary_endo'] = torch.as_tensor(endo, dtype=torch.float32, device=device)
        result['source_slice_indices'] = raw['source_slice_indices']
    return result


def require_shell(case):
    from .sampling import SCHEMA
    if case.get('schema') != SCHEMA:
        raise ValueError('Legacy epi-only cache is not a shell input. Rebuild with adapt_static_cache.py and prepare_cache.py; see docs/SHELL_DEPLOYMENT.md')


def select_view(case, points):
    n, p = case['q'].shape[:2]
    # Independently resample each plane; replacement allowed for small caches.
    index = torch.randint(p, (n, points), device=case['q'].device)
    row = torch.arange(n, device=index.device)[:, None]
    return case['q'][row, index], case['sdf2d'][row, index]


def gap_queries(case, count, eps=0.003):
    """Compatibility wrapper for shell queries, without pseudo targets."""
    from .sampling import eikonal_queries
    q = eikonal_queries(case, count)
    s = case['s']; pos = q @ case['axis']
    k = (torch.searchsorted(s, pos.contiguous(), right=True)-1).clamp(0, len(s)-2)
    return q, k, (pos-s[k])/(s[k+1]-s[k])


def validate_split(paths):
    """No same patient across splits, including ED/ES or anatomy-specific files."""
    sets = {}
    for name, entries in paths.items():
        ids = set()
        for path in entries:
            with np.load(path, allow_pickle=False) as f:
                ids.add(str(f['case_id']))
        sets[name] = ids
    names = list(sets)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if sets[a] & sets[b]:
                raise ValueError(f'Patient leakage: {a}/{b}: {sets[a] & sets[b]}')
