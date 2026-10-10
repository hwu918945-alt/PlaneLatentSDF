"""True zero-contour geometry; all measurements in aligned millimetres."""
import numpy as np
import torch
from skimage.measure import find_contours
from core.mesh import stack_basis
from core.model import leave_one_out_latent
from vendor.static_metrics import _slice_metrics


def slice_geometry(model, case, latent, plane, spacing_mm=1.0, chunk=8192):
    basis = stack_basis(case['axis'].detach().cpu().numpy())
    boundary = case['boundary'][plane].detach().cpu().numpy()
    uv = boundary @ basis[:, :2]
    step = spacing_mm/case['scale_mm']
    lower, upper = uv.min(0)-0.15, uv.max(0)+0.15
    sizes = np.ceil((upper-lower)/step).astype(int)+1
    axes = [lower[j]+np.arange(sizes[j])*step for j in range(2)]
    grid = np.stack(np.meshgrid(*axes, indexing='ij'), -1)
    pos = float(case['s'][plane])
    xyz = np.c_[grid.reshape(-1, 2), np.full(grid.size//2, pos)] @ basis.T
    values = []
    with torch.no_grad():
        for start in range(0, len(xyz), chunk):
            q = torch.as_tensor(xyz[start:start+chunk], dtype=torch.float32, device=latent.device)
            values.append(model.decoder(q, latent.expand(len(q), -1), case['phase']).cpu().numpy())
    field = np.concatenate(values).reshape(tuple(sizes))
    if not np.isfinite(field).all():
        raise FloatingPointError(f'Nonfinite evaluation: {case["case_id"]}')
    curves = find_contours(field, 0) if field.min()<0<field.max() else []
    segments, closed = [], []
    for curve in curves:
        coords = lower+curve*step
        world = np.c_[coords, np.full(len(coords), pos)] @ basis.T
        world = world*case['scale_mm']+case['center_mm']
        segments.extend(np.stack([world[:-1], world[1:]], 1))
        closed.append(np.linalg.norm(curve[0]-curve[-1])<1e-3)
    observed = boundary*case['scale_mm']+case['center_mm']
    origin = pos*basis[:, 2]*case['scale_mm']+case['center_mm']
    metrics = _slice_metrics(observed, np.asarray(segments), origin, basis[:, 0], basis[:, 1], 0.5, True) if segments else None
    valid = metrics is not None and all(closed)
    # Open/truncated predictions have undefined filled-region Dice.
    if metrics is not None and not all(closed):
        metrics['contour_dice'] = None
    area = abs(np.sum(uv[:,0]*np.roll(uv[:,1],-1)-uv[:,1]*np.roll(uv[:,0],-1)))/2*case['scale_mm']**2
    return dict(plane=plane, s_mm=pos*case['scale_mm'], valid=valid, metrics=metrics,
                observed_section_area_mm2=float(area),
                predicted_section_area_mm2=float((field<0).sum()*spacing_mm**2) if valid else None,
                predicted_components=len(curves), truncated=bool(curves and not all(closed))), field, lower, step


def case_geometry(model, case, spacing_mm=1.0):
    with torch.no_grad():
        z = model.encoder(case['q'], case['sdf2d'], case['phase'])
        zl = leave_one_out_latent(case['s'], z)
    direct, loo = [], []
    for i in range(len(z)):
        direct.append(slice_geometry(model, case, z[i:i+1], i, spacing_mm)[0])
        if 0 < i < len(z)-1:
            loo.append(slice_geometry(model, case, zl[i-1:i], i, spacing_mm)[0])
    return {'direct':direct, 'loo':loo}


def aggregate_geometry(cases):
    result = {}
    for mode in ['direct', 'loo']:
        summaries = []
        for case in cases:
            rows = case[mode]
            values = {}
            for metric in ['mcd_mm','hd95_mm','contour_dice']:
                valid = [r['metrics'][metric] for r in rows if r['metrics'] and r['metrics'][metric] is not None]
                values[metric] = float(np.mean(valid)) if valid else None
            values['coverage'] = sum(r['valid'] for r in rows)/len(rows) if rows else 0.
            # Missing or truncated levels receive an explicit selection penalty.
            values['selection_penalty'] = np.mean([r['metrics']['mcd_mm'] if r['valid'] else 50. for r in rows]) if rows else 50.
            summaries.append(values)
        for metric in ['mcd_mm','hd95_mm','contour_dice','coverage','selection_penalty']:
            valid = [r[metric] for r in summaries if r[metric] is not None]
            result[f'{mode}_{metric}'] = float(np.mean(valid)) if valid else None
    return result
