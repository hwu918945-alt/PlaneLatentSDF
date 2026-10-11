"""Same class-area-uniform shell pools with bounded class proposal boxes.

Region0 proposals use that plane's Epi bbox, region1 its Endo bbox, region2
uses the unchanged common observed ROI+margin. Accepted class-conditional
densities equal uniform proposals over the common ROI, without wasting
positive proposals outside the containing polygon's box.
"""
import hashlib,json,time
from pathlib import Path
import numpy as np
from .area_pool import PlaneClassifier,digest
from .data import load_case
from .inference import bounds


def build_pool(case,observed,output,capacities=(131072,65536,65536),seed=19,generation=0,
               margin_mm=20.,boundary_band_mm=.05,near_mm=3.,max_candidates=20000000):
    start=time.perf_counter();out=Path(output)
    if out.exists():raise FileExistsError(out)
    c=load_case(case,observed=observed);lower,upper,basis=bounds(c,margin_mm);scale=c['scale_mm']
    rng=np.random.default_rng(np.random.SeedSequence([seed,48191,generation]));allq=[];allnear=[];audit=[]
    for i,pos in enumerate(c['s'].numpy()):
        epi=c['boundary'][i].numpy()@basis[:,:2]*scale;endo=c['boundary_endo'][i].numpy()@basis[:,:2]*scale
        classifier=PlaneClassifier(epi,endo)
        boxes=[(epi.min(0)/scale,epi.max(0)/scale),(endo.min(0)/scale,endo.max(0)/scale),(lower[:2],upper[:2])]
        parts=[];nears=[];proposals=[]
        for r,cap in enumerate(capacities):
            remaining=cap;items=[];nearitems=[];proposed=0
            while remaining:
                size=65536;proposed+=size
                if proposed>max_candidates:raise ValueError(f'Plane {observed[i]} region {r}: insufficient accepted area')
                uv=rng.uniform(*boxes[r],(size,2));q=(uv@basis[:,:2].T+float(pos)*basis[:,2]).astype(np.float32)
                region,valid,near=classifier.classify(q.astype(np.float64)@basis[:,:2]*scale,boundary_band_mm,near_mm)
                ix=np.flatnonzero(valid & (region==r))[:remaining]
                if len(ix):items.append(q[ix]);nearitems.append(near[ix]);remaining-=len(ix)
            parts.append(np.concatenate(items));nears.append(np.concatenate(nearitems));proposals.append(proposed)
        q=np.concatenate(parts);near=np.concatenate(nears)
        if len(np.unique(q,axis=0))!=sum(capacities):raise ValueError('Duplicate shell area points')
        if np.max(abs(q@c['axis'].numpy()-pos))>1e-5:raise ValueError('Plane mismatch')
        allq.append(q);allnear.append(near);audit.append(dict(index=observed[i],proposed_by_region=proposals,pools=list(capacities),
            near_counts=[int(n.sum()) for n in nears]))
    q=np.stack(allq);near=np.stack(allnear);region=np.tile(np.repeat(np.arange(3,dtype=np.int8),capacities),(len(observed),1))
    meta=dict(schema='plane_auto_occ_area_uniform_v2',source_sha256=digest(case),observed=list(observed),seed=seed,generation=generation,
        capacities=list(capacities),roi_stack_mm=(np.stack((lower,upper))*scale).tolist(),margin_mm=margin_mm,
        boundary_band_mm=boundary_band_mm,near_mm=near_mm,
        contour_sha256=hashlib.sha256(c['boundary'].numpy().tobytes()+c['boundary_endo'].numpy().tobytes()).hexdigest(),
        hidden_geometry_used=False,pool_sha256=hashlib.sha256(q.tobytes()+region.tobytes()+near.tobytes()).hexdigest(),planes=audit,
        proposal='class containing bbox for MYO/cavity; unchanged observed-only common ROI for background',
        conditional_area_density='uniform in each true region, excluding boundary band',build_seconds=time.perf_counter()-start)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,q=q,region=region,near=near,occ=(region==0).astype(np.float32),metadata_json=np.array(json.dumps(meta)))
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2));print(json.dumps(meta),flush=True);return meta
