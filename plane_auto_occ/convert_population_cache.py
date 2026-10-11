"""Bounded-memory shell conversion using exact segment-band geometry.

Same authoritative shell sign and class semantics as convert_cache. The KD-tree
omits only segments proven too far to affect either rejection/near threshold.
"""
from pathlib import Path
import numpy as np
from .area_pool import PlaneClassifier,digest
from .geometry import stack_basis


def convert(src,dst,boundary_band_mm=.05,near_mm=3.):
    src=Path(src);dst=Path(dst)
    if dst.exists() or src.resolve()==dst.resolve():raise FileExistsError(dst)
    with np.load(src,allow_pickle=False) as z:
        keys=['q','sdf2d','boundary','boundary_endo','s','axis','center_mm','scale_mm','phase','case_id']
        raw={k:np.array(z[k]) for k in keys}
        sampling=np.array(z['sampling_source']) if 'sampling_source' in z else None
        source_ids=np.array(z['source_slice_indices']) if 'source_slice_indices' in z else np.arange(len(raw['s']))
    scale=float(raw['scale_mm']);basis=stack_basis(raw['axis']);regions=[];nears=[];occ=[]
    for i,q in enumerate(raw['q']):
        classifier=PlaneClassifier(raw['boundary'][i]@basis[:,:2]*scale,raw['boundary_endo'][i]@basis[:,:2]*scale)
        r,valid,near=classifier.classify(q@basis[:,:2]*scale,boundary_band_mm,near_mm)
        old=raw['sdf2d'][i]
        if not np.isfinite(old).all():raise ValueError('Nonfinite source SDF')
        if sampling is not None:valid &= (sampling[i]!=0)&(sampling[i]!=2)
        valid &= abs(old)*scale>boundary_band_mm
        if np.any(((old<0)!=(r==0)) & valid):raise ValueError(f'Source shell sign disagreement: {src}, {i}')
        r[~valid]=-1
        if any(not np.any(r==k) for k in (0,1,2)):raise ValueError(f'Missing source class: {src}, {i}')
        regions.append(r.astype(np.int64));nears.append(near);occ.append((old<0).astype(np.float32))
    output={k:v for k,v in raw.items() if k!='sdf2d'}
    output.update(occ=np.array(occ),region=np.array(regions),near=np.array(nears),source_slice_indices=source_ids,
        schema_version=np.array('plane_auto_occ_v1'),source_sha256=np.array(digest(src)))
    dst.parent.mkdir(parents=True,exist_ok=True);temporary=dst.with_suffix('.tmp.npz')
    np.savez_compressed(temporary,**output);temporary.replace(dst)
