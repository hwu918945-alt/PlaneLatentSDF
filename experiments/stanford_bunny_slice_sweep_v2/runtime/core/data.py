"""NPZ integration contract; deliberately independent of unknown cache internals."""
from pathlib import Path
import numpy as np
import torch


def polygon_sdf(points, polygon):
    """Exact segment distance + ray crossing sign for ONE simple filled polygon.
    Negative inside. Points exactly on a segment have zero distance.
    TODO(AGENT): epi or cavity is a filled region; do NOT use a myocardium ring here.
    """
    p = np.asarray(points, dtype=np.float64)
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
    return result


def select_view(case, points):
    n, p = case['q'].shape[:2]
    # Independently resample each plane; replacement allowed for small caches.
    index = torch.randint(p, (n, points), device=case['q'].device)
    row = torch.arange(n, device=index.device)[:, None]
    return case['q'][row, index], case['sdf2d'][row, index]


def gap_queries(case, count, eps=0.003):
    """Proposal from roughly interpolated contours, plus broad planar Gaussian.
    No target sdf and no midpoint encoder. Only observed planes anchor intervals.
    TODO(AGENT): optional proposal alignment may rotate the ordered starting index;
    NEVER train against the proposal as a pseudo ground-truth contour.
    """
    s, b, axis = case['s'], case['boundary'], case['axis']
    valid = torch.nonzero(s[1:] - s[:-1] > 4 * eps).flatten()
    if not len(valid):
        raise ValueError('No gap large enough for geometry finite differences')
    k = valid[torch.randint(len(valid), (count,), device=s.device)]
    clearance = 1.5 * eps / (s[k + 1] - s[k])
    a = clearance + torch.rand(count, device=s.device) * (1 - 2 * clearance)
    j = torch.randint(b.shape[1], (count,), device=s.device)
    proposal = (1 - a[:, None]) * b[k, j] + a[:, None] * b[k + 1, j]
    noise = torch.randn_like(proposal) * 0.06  # 6 mm at scale=100 mm.
    noise -= (noise * axis).sum(-1, keepdim=True) * axis
    q = proposal + noise
    # Include broad uniform lateral samples in the box of observed contours.
    uniform = b.amin((0, 1)) + torch.rand_like(q) * (b.amax((0, 1)) - b.amin((0, 1)))
    pos = (1 - a) * s[k] + a * s[k + 1]
    uniform += (pos - uniform @ axis)[:, None] * axis
    mask = torch.rand(count, device=s.device) < 0.25
    q[mask] = uniform[mask]
    return q, k, a


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
