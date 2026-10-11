"""Read shell cache; write new occupancy cache without modifying source."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from .geometry import stack_basis, contour_labels


def convert(src, dst, boundary_band_mm=0.05, near_mm=3.0):
    src, dst = Path(src), Path(dst)
    if src.resolve() == dst.resolve() or dst.exists(): raise FileExistsError(dst)
    with np.load(src, allow_pickle=False) as f:
        keys = ['q','sdf2d','boundary','boundary_endo','s','axis','center_mm','scale_mm','phase','case_id']
        if any(k not in f for k in keys):
            raise ValueError(f'{src}: need true myocardium shell cache with boundary_endo, not epi-only cache')
        raw = {k: np.array(f[k]) for k in keys}
        source = np.array(f['sampling_source']) if 'sampling_source' in f else None
        source_ids = np.array(f['source_slice_indices']) if 'source_slice_indices' in f else np.arange(len(raw['s']))
    scale = float(raw['scale_mm'])
    if not np.isfinite(scale) or scale <= 0: raise ValueError('Invalid scale_mm')
    if raw['sdf2d'].shape != raw['q'].shape[:2]: raise ValueError('Invalid shell sdf2d shape')
    basis = stack_basis(raw['axis'])
    regions, near_rows, occ_rows, stats = [], [], [], []
    for i, q in enumerate(raw['q']):
        r, de, di = contour_labels(q, raw['boundary'][i], raw['boundary_endo'][i], basis)
        distance = np.minimum(np.abs(de), np.abs(di))*scale
        valid = distance > boundary_band_mm
        if source is not None: valid &= (source[i] != 0) & (source[i] != 2)
        old = raw['sdf2d'][i]
        if not np.isfinite(old).all(): raise ValueError('Nonfinite shell labels')
        if np.any(((old < 0) != (r == 0)) & valid & (np.abs(old)*scale > boundary_band_mm)):
            raise ValueError(f'{src}: plane {i}: shell sign disagrees with epi/endo polygons')
        # Label source is existing shell sign. Polygons classify negative subregions
        # and audit semantics, rather than treating a filled Epi envelope as myocardium.
        occ = (old < 0).astype(np.float32)
        valid &= np.abs(old)*scale > boundary_band_mm
        r[~valid] = -1
        counts = {str(k): int((r == k).sum()) for k in (0,1,2)}
        if min(counts.values()) == 0: raise ValueError(f'{src}: plane {i}: missing class pool {counts}')
        regions.append(r); occ_rows.append(occ); near_rows.append(distance <= near_mm)
        stats.append(counts)
    out = {k: v for k,v in raw.items() if k != 'sdf2d'}
    out.update(occ=np.asarray(occ_rows,np.float32), region=np.asarray(regions,np.int64),
               near=np.asarray(near_rows,bool), source_slice_indices=source_ids,
               schema_version=np.array('plane_auto_occ_v1'),
               source_sha256=np.array(hashlib.sha256(src.read_bytes()).hexdigest()))
    dst.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst, **out)
    return dict(source=str(src), output=str(dst), pools=stats)


def main():
    p=argparse.ArgumentParser(); p.add_argument('--input-root', required=True); p.add_argument('--output-root', required=True)
    p.add_argument('--boundary-band-mm',type=float,default=0.05); p.add_argument('--near-mm',type=float,default=3.)
    a=p.parse_args(); src=Path(a.input_root).resolve(); dst=Path(a.output_root).resolve()
    if src == dst or src in dst.parents or dst in src.parents: raise ValueError('Use disjoint cache roots')
    if a.boundary_band_mm < 0 or a.near_mm <= a.boundary_band_mm: raise ValueError('Invalid sampling bands')
    files=sorted(src.rglob('*.npz'))
    if not files: raise ValueError('No source NPZ')
    report=[convert(f,dst/f.relative_to(src),a.boundary_band_mm,a.near_mm) for f in files]
    (dst/'conversion_audit.json').write_text(json.dumps(report,indent=2))
    print(f'Converted {len(report)} files; original cache preserved')


if __name__ == '__main__': main()
