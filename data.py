from pathlib import Path
import numpy as np
import torch


def load_case(path, device='cpu', observed=None):
    with np.load(path, allow_pickle=False) as f:
        required = ['q', 'occ', 'region', 'near', 'boundary', 'boundary_endo', 's', 'axis',
                    'center_mm', 'scale_mm', 'phase', 'case_id', 'schema_version']
        if any(k not in f for k in required):
            raise ValueError(f'{path}: run convert_cache first; missing occupancy fields')
        r = {k: np.array(f[k]) for k in required}
    if str(r['schema_version']) != 'plane_auto_occ_v1':
        raise ValueError('Unsupported cache version')
    q = r['q']; s = r['s']; axis = r['axis']
    if q.ndim != 3 or q.shape[-1] != 3 or len(q) != len(s):
        raise ValueError('Invalid q/s')
    for k in ('occ', 'region', 'near'):
        if r[k].shape != q.shape[:2]: raise ValueError(f'Invalid {k} shape')
    for k in ('q','s','axis','boundary','boundary_endo','center_mm','scale_mm'):
        if not np.isfinite(r[k]).all(): raise ValueError(f'Nonfinite {k}')
    if axis.shape != (3,) or abs(np.linalg.norm(axis)-1) > 1e-4:
        raise ValueError('Invalid axis')
    if np.any(np.diff(s) <= 1e-5) or np.max(np.abs(q @ axis-s[:, None])) > 1e-4:
        raise ValueError('Unsorted or nonparallel planes')
    for k in ('boundary','boundary_endo'):
        if r[k].ndim != 3 or r[k].shape[0] != len(s) or r[k].shape[-1] != 3:
            raise ValueError(f'Invalid {k}')
        if np.max(np.abs(r[k] @ axis-s[:, None])) > 1e-4: raise ValueError('Contour plane mismatch')
    if not np.isin(r['occ'], [0,1]).all() or not np.isin(r['region'], [-1,0,1,2]).all():
        raise ValueError('Invalid class labels')
    valid = r['region'] >= 0
    if not np.array_equal(r['occ'][valid], (r['region'][valid] == 0).astype(np.float32)):
        raise ValueError('Myocardium must be 1; cavity/background 0')
    if float(r['scale_mm']) <= 0 or r['center_mm'].shape != (3,) or int(r['phase']) not in (0,1):
        raise ValueError('Invalid normalization/phase')
    ids = np.arange(len(s)) if observed is None else np.asarray(observed, dtype=np.int64)
    if len(ids) < 2 or np.any(np.diff(ids) <= 0) or ids[0] < 0 or ids[-1] >= len(s):
        raise ValueError('observed must contain >=2 sorted unique in-range indices')
    result = {k: torch.as_tensor(r[k][ids], device=device,
              dtype=(torch.int64 if k == 'region' else torch.bool if k == 'near' else torch.float32))
              for k in ('q','occ','region','near','boundary','boundary_endo','s')}
    result.update(axis=torch.as_tensor(axis, dtype=torch.float32, device=device),
                  phase=int(r['phase']), case_id=str(r['case_id']),
                  center_mm=r['center_mm'], scale_mm=float(r['scale_mm']),
                  path=str(Path(path).resolve()), observed=ids.tolist(), total_planes=len(s))
    return result


def sample_plane(c, i, count):
    if count < 4 or count % 4: raise ValueError('points must be divisible by 4')
    ids, fallbacks = [], 0
    for region, budget in ((0,count//2),(1,count//4),(2,count//4)):
        all_ids = torch.nonzero(c['region'][i] == region).flatten()
        if not len(all_ids): raise ValueError(f'{c["path"]} plane {i}: missing region {region}')
        for near, n in ((True,budget//2),(False,budget-budget//2)):
            pool = all_ids[c['near'][i, all_ids].bool() == near]
            if not len(pool): pool = all_ids; fallbacks += 1
            ids.append(pool[torch.randint(len(pool), (n,), device=pool.device)])
    ids = torch.cat(ids)
    return c['q'][i,ids].float(), c['occ'][i,ids].float(), fallbacks


def records(data_root=None, case=None, manifest=None, observed=None):
    import json
    if sum(v is not None for v in (data_root, case, manifest)) != 1:
        raise ValueError('Choose exactly one of --data-root, --case, --manifest')
    if manifest:
        p = Path(manifest).resolve()
        rows = json.loads(p.read_text())['records']
        return [(str((p.parent/row['path']).resolve()), row.get('observed')) for row in rows]
    if case: return [(case, observed)]
    paths = sorted(Path(data_root).glob('*.npz'))
    if not paths: raise ValueError('No NPZ files in requested split directory')
    return [(str(p), observed) for p in paths]
