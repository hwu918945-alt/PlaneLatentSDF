"""Read only SAX epi contours from the retained static alignment cache."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np


def ordered_polygon(uv):
    uv = np.asarray(uv, np.float64)
    uv = uv[np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1) > 1e-7]
    if len(uv) < 3:
        raise ValueError('Degenerate polygon')
    a, b = uv, np.roll(uv, -1, axis=0)
    area = np.sum(a[:, 0]*b[:, 1] - b[:, 0]*a[:, 1]) / 2
    if abs(area) < 1e-4:
        raise ValueError('Degenerate polygon area')
    # Reject nonadjacent segment crossings rather than silently filling them.
    delta = b-a
    def cross(x, y):
        return x[..., 0]*y[..., 1]-x[..., 1]*y[..., 0]
    denominator = cross(delta[:, None], delta[None])
    displacement = a[None]-a[:, None]
    safe = np.where(abs(denominator) > 1e-10, denominator, 1)
    t = cross(displacement, delta[None])/safe
    u = cross(displacement, delta[:, None])/safe
    crossing = (abs(denominator)>1e-10)&(t>1e-7)&(t<1-1e-7)&(u>1e-7)&(u<1-1e-7)
    if crossing.any():
        raise ValueError('Self-intersecting polygon')
    if area < 0:
        uv = uv[::-1]
    # A consistent positive-u angular start; no assumed anatomical correspondence.
    relative = uv-uv.mean(0)
    start = np.argmin(abs(np.arctan2(relative[:, 1], relative[:, 0])))
    return np.roll(uv, -int(start), axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', default='data/aligned_sax_manifest.json')
    args = p.parse_args()
    source = Path(args.source).resolve()
    manifest_path = source/'data/static_endpoint_manifest.json'
    original = json.loads(manifest_path.read_text())
    records, reports = [], []
    for row in original['records']:
        if row['split'] == 'stress':
            continue
        cid = str(row['case_id'])
        path = source/'data/cache'/cid/'endpoint_case.npz'
        with np.load(path, allow_pickle=False) as f:
            # No masks, training_myo_views, LAX, template, or motion data is read.
            keys = ['contours', 'contour_valid', 'contour_closed', 'view_ids',
                    'plane_origins', 'plane_normals', 'plane_u', 'plane_v',
                    'center', 'scale', 'metadata']
            cache = {k: np.array(f[k]) for k in keys}
        metadata = json.loads(str(cache['metadata'].item()))
        if metadata['case_id'] != cid or metadata['split'] != row['split']:
            raise ValueError(f'Identity mismatch: {cid}')
        center, scale = cache['center'].astype(float), float(cache['scale'])
        for ph, phase in enumerate(['ED', 'ES']):
            ids = np.flatnonzero((cache['view_ids']==0)&cache['contour_closed']&cache['contour_valid'][ph, :, 0])
            if len(ids) < 3:
                raise ValueError(f'{cid}/{phase}: need three real planes for LOO')
            planes, errors = [], []
            for i in ids:
                origin = cache['plane_origins'][ph, i].astype(float)*scale+center
                eu = cache['plane_u'][ph, i].astype(float)
                ev = cache['plane_v'][ph, i].astype(float)
                eu /= np.linalg.norm(eu)
                ev -= (eu@ev)*eu; ev /= np.linalg.norm(ev)
                normal = np.cross(eu, ev)
                expected = cache['plane_normals'][ph, i].astype(float)
                expected /= np.linalg.norm(expected)
                if abs(normal@expected) < 1-1e-6:
                    raise ValueError(f'Plane basis mismatch: {cid}/{phase}/{i}')
                points = cache['contours'][ph, i, 0].astype(float)*scale+center
                residual = (points-origin)@normal
                error = float(abs(residual).max()); errors.append(error)
                if error > 0.05:
                    raise ValueError(f'Unexpected out-of-plane error: {cid}/{phase}/{i}: {error}')
                uv = ordered_polygon(np.stack([(points-origin)@eu, (points-origin)@ev], -1))
                planes.append(dict(origin_mm=origin.tolist(), u=eu.tolist(), v=ev.tolist(),
                                   contour_uv_mm=uv.tolist(), source_slice_index=int(i),
                                   quantization_plane_residual_mm=error))
            records.append(dict(case_id=cid, phase=phase, split=row['split'], anatomy='LV_epi_envelope',
                                planes=planes, canonical_transform=row['canonical_transform'],
                                case_orientation=row['case_orientation'], source_cache=str(path)))
            reports.append(dict(case_id=cid, phase=phase, split=row['split'], planes=len(planes),
                                max_quantization_plane_residual_mm=max(errors)))
    counts = Counter(r['split'] for r in records)
    if counts != {'train':570, 'val':72, 'test':72}:
        raise ValueError(f'Fixed split mismatch: {counts}')
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(records=records, anatomy='LV_epi_envelope', scale_mm=100,
                                   source_manifest=str(manifest_path)), separators=(',', ':')))
    audit = dict(source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 phase_file_counts=dict(counts), patient_counts={k:v//2 for k,v in counts.items()},
                 excluded_stress_patients=3, field_mapping='aligned_mm = contours * static.scale + static.center',
                 plane_basis='unit-normalized plane_u/plane_v; spacing already in 3D contours',
                 quantization_projection_max_mm=max(r['max_quantization_plane_residual_mm'] for r in reports),
                 common_parallel_approximation=False, records=reports)
    out.with_name('alignment_adapter_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k!='records'}, indent=2))


if __name__ == '__main__':
    main()
