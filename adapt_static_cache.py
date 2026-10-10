"""Read SAX epi/endo; verify channel semantics by strict geometric nesting."""
import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
import numpy as np
from core.geometry import validate_shell


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', required=True)
    p.add_argument('--output', default='data/aligned_sax_shell_manifest.json')
    p.add_argument('--allow-subset', action='store_true', help='Testing only: permit a partial source manifest')
    args = p.parse_args()
    source = Path(args.source).resolve()
    manifest_path = source/'data/static_endpoint_manifest.json'
    original = json.loads(manifest_path.read_text())
    records, reports, excluded = [], [], []
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
        if cache['contours'].ndim != 5 or cache['contours'].shape[2] != 2 or metadata.get('input_contract') != 'endpoint epi+endo contours plus plane geometry':
            raise ValueError(f'Unknown channel contract: {cid}')
        if metadata['case_id'] != cid or metadata['split'] != row['split']:
            raise ValueError(f'Identity mismatch: {cid}')
        center, scale = cache['center'].astype(float), float(cache['scale'])
        for ph, phase in enumerate(['ED', 'ES']):
            ids = np.flatnonzero((cache['view_ids']==0)&cache['contour_closed']&cache['contour_valid'][ph].all(-1))
            if len(ids) < 3:
                raise ValueError(f'{cid}/{phase}: need three real planes for LOO')
            planes, errors = [], []
            sax_ids = np.flatnonzero((cache['view_ids']==0)&cache['contour_closed'])
            for missing in sorted(set(sax_ids)-set(ids)):
                excluded.append(dict(case_id=cid, phase=phase, source_slice_index=int(missing), reason='missing_or_invalid_epi_endo', valid_channels=cache['contour_valid'][ph, missing].tolist()))
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
                polygons, residuals = [], []
                for channel in (0, 1):
                    points = cache['contours'][ph, i, channel].astype(float)*scale+center
                    error = float(abs((points-origin)@normal).max()); residuals.append(error)
                    if error > 0.05:
                        raise ValueError(f'Unexpected out-of-plane error: {cid}/{phase}/{i}/{channel}: {error}')
                    polygons.append(np.stack([(points-origin)@eu, (points-origin)@ev], -1))
                try:
                    # Metadata describes epi+endo; geometric containment verifies
                    # channel 0 as outer and 1 as inner rather than trusting index.
                    epi, endo = validate_shell(*polygons)
                except ValueError as error:
                    excluded.append(dict(case_id=cid, phase=phase, source_slice_index=int(i), reason=str(error)))
                    continue
                error = max(residuals); errors.append(error)
                planes.append(dict(origin_mm=origin.tolist(), u=eu.tolist(), v=ev.tolist(),
                                   epi_uv_mm=epi.tolist(), endo_uv_mm=endo.tolist(), source_slice_index=int(i),
                                   quantization_plane_residual_mm=error))
            if len(planes) < 3:
                raise ValueError(f'{cid}/{phase}: fewer than three valid nested shell planes; inspect exclusions: {excluded[-12:]}')
            records.append(dict(case_id=cid, phase=phase, split=row['split'], anatomy='LV_myocardium_shell',
                                planes=planes, canonical_transform=row['canonical_transform'],
                                case_orientation=row['case_orientation'], source_cache=str(path)))
            reports.append(dict(case_id=cid, phase=phase, split=row['split'], planes=len(planes),
                                max_quantization_plane_residual_mm=max(errors)))
    counts = Counter(r['split'] for r in records)
    if not args.allow_subset and counts != {'train':570, 'val':72, 'test':72}:
        raise ValueError(f'Fixed split mismatch: {counts}')
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(records=records, anatomy='LV_myocardium_shell', scale_mm=100,
                                   source_manifest=str(manifest_path)), separators=(',', ':')))
    audit = dict(source_manifest_sha256=hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                 phase_file_counts=dict(counts), patient_counts={k:v//2 for k,v in counts.items()},
                 excluded_stress_patients=3, field_mapping='aligned_mm = contours * static.scale + static.center',
                 plane_basis='unit-normalized plane_u/plane_v; spacing already in 3D contours',
                 quantization_projection_max_mm=max(r['max_quantization_plane_residual_mm'] for r in reports),
                 common_parallel_approximation=False, channel_mapping={'epi':0, 'endo':1},
                 channel_evidence='endpoint epi+endo metadata and per-plane strict continuous polygon containment',
                 excluded_planes=excluded, records=reports)
    out.with_name(out.stem+'_audit.json').write_text(json.dumps(audit, indent=2))
    print(json.dumps({k:v for k,v in audit.items() if k not in ('records','excluded_planes')}, indent=2))


if __name__ == '__main__':
    main()
