"""Binary solids from separate closed rings, with even/odd holes and components.

The two negative buckets are exchangeable outside samples. With the unchanged
0.5/0.25/0.25 BCE weights this is 0.5 inside + 0.5 outside BCE.
No artificial endocardium, joining segments, or mesh supervision is introduced.
"""
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from matplotlib.path import Path as PolygonPath
from scipy.spatial import cKDTree
from shapely.geometry import Polygon, GeometryCollection
from .area_pool import PlaneClassifier, digest


class SolidClassifier:
    band_distance = PlaneClassifier.band_distance

    def __init__(self, rings):
        self.paths=[]; self.edges=[]; self.rings=[]
        shape=GeometryCollection()
        for ring in rings:
            p=np.asarray(ring,np.float64)
            p=p[np.linalg.norm(p-np.roll(p,1,axis=0),axis=1)>1e-10]
            poly=Polygon(p)
            if not poly.is_valid or poly.area<=0: raise ValueError('Invalid closed solid ring')
            # Proper sections have nested/disjoint boundaries, never crossing ones.
            for earlier in self.rings:
                if poly.boundary.intersects(Polygon(earlier).boundary):
                    raise ValueError('Intersecting solid rings')
            self.rings.append(p); shape=shape.symmetric_difference(poly)
            self.paths.append(PolygonPath(np.vstack([p,p[0]]),closed=True))
            d=np.roll(p,-1,axis=0)-p
            self.edges.append((p,d,(d*d).sum(1),cKDTree(p+d/2),np.linalg.norm(d,axis=1).max()/2))
        if not shape.is_valid or shape.is_empty: raise ValueError('Invalid solid section')
        self.shape=shape
        components=[shape] if shape.geom_type=='Polygon' else list(shape.geoms)
        self.component_boxes=[]; self.component_paths=[]
        for component in components:
            if component.geom_type!='Polygon': raise ValueError('Nonpolygon component')
            self.component_boxes.append(component.bounds)
            self.component_paths.append([PolygonPath(np.asarray(component.exterior.coords),closed=True)]+
                [PolygonPath(np.asarray(h.coords),closed=True) for h in component.interiors])
        self.component_boxes=np.array(self.component_boxes)
        widths=self.component_boxes[:,2:]-self.component_boxes[:,:2]
        self.box_prob=widths.prod(1)/widths.prod(1).sum()

    def classify(self,points,band=.05,near_mm=3.):
        inside=np.zeros(len(points),bool)
        for path in self.paths: inside ^= path.contains_points(points)
        distance=np.minimum.reduce([self.band_distance(points,e,max(band,near_mm)) for e in self.edges])
        return np.where(inside,0,1).astype(np.int8),distance>band,distance<=near_mm

    def positive_proposals(self,rng,size):
        # Select boxes by box area; accept only in the selected component.
        # The accepted density is constant over the disjoint filled components,
        # including holes. This avoids enormous ROI rejection at tiny ear tips.
        ids=rng.choice(len(self.component_boxes),size,p=self.box_prob)
        box=self.component_boxes[ids]
        uv=box[:,:2]+rng.random((size,2))*(box[:,2:]-box[:,:2])
        own=np.zeros(size,bool)
        for i,paths in enumerate(self.component_paths):
            idx=np.flatnonzero(ids==i); mask=paths[0].contains_points(uv[idx])
            for hole in paths[1:]: mask &= ~hole.contains_points(uv[idx])
            own[idx]=mask
        return uv,own


def convert(source, contours, output):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    planes=json.loads(Path(contours).read_text())['planes']
    with np.load(source,allow_pickle=False) as z:
        raw={k:np.array(z[k]) for k in ('q','boundary','s','axis','center_mm','scale_mm','phase','case_id')}
    if len(planes)!=len(raw['s']) or not np.allclose(raw['axis'],[0,0,1]):
        raise ValueError('Expected world-XY section rings on a Z stack')
    center=raw['center_mm']; scale=float(raw['scale_mm']); points=[]; offsets=[0]; plane_offsets=[0]
    labels=[]; regions=[]; nears=[]
    for i,plane in enumerate(planes):
        if abs((plane['height_mm']-center[2])/scale-raw['s'][i])>1e-5: raise ValueError('Section height mismatch')
        classifier=SolidClassifier(plane['rings_uv_mm'])
        r,valid,near=classifier.classify(raw['q'][i,:,:2]*scale+center[:2])
        labels.append((r==0).astype(np.float32)); r[~valid]=-1; regions.append(r); nears.append(near)
        for ring in classifier.rings:
            points.extend(ring); offsets.append(len(points))
        plane_offsets.append(len(offsets)-1)
    raw.update(occ=np.array(labels),region=np.array(regions),near=np.array(nears),
        ring_points_mm=np.array(points,np.float64),ring_offsets=np.array(offsets,np.int64),
        plane_ring_offsets=np.array(plane_offsets,np.int64),schema_version=np.array('plane_auto_occ_solid_v1'),
        source_sha256=np.array(digest(source)),contours_sha256=np.array(digest(contours)))
    output.parent.mkdir(parents=True,exist_ok=True); np.savez_compressed(output,**raw)
    # Re-read through the validation path, including all ring and plane contracts.
    load_case(output)
    return dict(source_sha256=digest(source),contours_sha256=digest(contours),output_sha256=digest(output),
        planes=len(planes),rings_per_plane=np.diff(plane_offsets).tolist(),schema='plane_auto_occ_solid_v1')


def load_case(path,device='cpu',observed=None):
    with np.load(path,allow_pickle=False) as z:
        raw={k:np.array(z[k]) for k in z.files}
    if str(raw['schema_version'])!='plane_auto_occ_solid_v1': raise ValueError('Wrong solid schema')
    for k in ('q','boundary','s','axis','center_mm','scale_mm','ring_points_mm'):
        if not np.isfinite(raw[k]).all(): raise ValueError('Nonfinite solid cache')
    if not np.allclose(raw['axis'],[0,0,1]) or float(raw['scale_mm'])<=0 or int(raw['phase'])!=0:
        raise ValueError('Invalid solid normalization/phase')
    s=raw['s']; n=len(s); ids=np.arange(n) if observed is None else np.array(observed,dtype=int)
    if len(ids)<2 or np.any(np.diff(ids)<=0) or ids[0]<0 or ids[-1]>=n: raise ValueError('Invalid solid observed indices')
    if np.any(np.diff(s)<=1e-5) or np.max(abs(raw['q']@raw['axis']-s[:,None]))>1e-4: raise ValueError('Invalid solid planes')
    ro=raw['ring_offsets']; po=raw['plane_ring_offsets']; rp=raw['ring_points_mm']
    if ro[0]!=0 or ro[-1]!=len(rp) or np.any(np.diff(ro)<3) or po[0]!=0 or po[-1]!=len(ro)-1 or len(po)!=n+1 or np.any(np.diff(po)<1):
        raise ValueError('Invalid packed rings')
    rings=[[rp[ro[j]:ro[j+1]] for j in range(po[i],po[i+1])] for i in ids]
    for r in rings: SolidClassifier(r)
    q=raw['q']; valid=raw['region']>=0
    if raw['occ'].shape!=q.shape[:2] or not np.array_equal(raw['occ'][valid],(raw['region'][valid]==0).astype(np.float32)):
        raise ValueError('Invalid solid occupancy')
    result={k:torch.as_tensor(raw[k][ids],device=device,dtype=(torch.int64 if k=='region' else torch.bool if k=='near' else torch.float32))
        for k in ('q','occ','region','near','boundary','s')}
    # Only observed rings enter bounds or training, regardless of extra cached planes.
    geometry=np.concatenate([np.c_[(np.concatenate(r)-raw['center_mm'][:2])/float(raw['scale_mm']),
        np.full(sum(len(x) for x in r),s[i])] for r,i in zip(rings,ids)])
    result.update(axis=torch.tensor(raw['axis'],device=device,dtype=torch.float32),phase=0,case_id=str(raw['case_id']),
        center_mm=raw['center_mm'],scale_mm=float(raw['scale_mm']),path=str(Path(path).resolve()),
        observed=ids.tolist(),total_planes=n,schema='plane_auto_occ_solid_v1',solid_rings_mm=rings,
        geometry_points=torch.tensor(geometry,device=device,dtype=torch.float32),
        region_names=['inside','outside_a','outside_b'])
    return result


def build_pool(case,observed,output,capacities=(131072,65536,65536),seed=19,generation=0,
               margin_mm=20.,boundary_band_mm=.05,near_mm=3.,max_candidates=20000000):
    from .inference import bounds
    start=time.perf_counter(); output=Path(output)
    if output.exists(): raise FileExistsError(output)
    c=load_case(case,observed=observed); scale=c['scale_mm']; center=c['center_mm']
    lower,upper,basis=bounds(c,margin_mm)
    box=np.stack([lower,upper])@basis.T
    # stack basis for Z is an XY rotation; actual world bbox from observed rings.
    allpoints=np.concatenate([np.concatenate(r) for r in c['solid_rings_mm']])
    xylo=allpoints.min(0)-margin_mm; xyhi=allpoints.max(0)+margin_mm
    rng=np.random.default_rng(np.random.SeedSequence([seed,48191,generation]))
    allq=[]; allnear=[]; audit=[]
    for i,rings in enumerate(c['solid_rings_mm']):
        classifier=SolidClassifier(rings); parts=[]; near_parts=[]; proposals=[]
        for r,cap in enumerate(capacities):
            remaining=cap; items=[]; near_items=[]; proposed=0
            while remaining:
                size=min(65536,max(4096,remaining*2)); proposed+=size
                if proposed>max_candidates: raise ValueError(f'Solid plane {observed[i]} region {r}: insufficient area')
                if r==0: uv,own=classifier.positive_proposals(rng,size)
                else: uv=rng.uniform(xylo,xyhi,(size,2)); own=np.ones(size,bool)
                q=np.c_[(uv-center[:2])/scale,np.full(size,float(c['s'][i]))].astype(np.float32)
                region,valid,near=classifier.classify(q[:,:2].astype(np.float64)*scale+center[:2],boundary_band_mm,near_mm)
                ix=np.flatnonzero(own & valid & ((region==0) if r==0 else (region!=0)))[:remaining]
                if len(ix): items.append(q[ix]); near_items.append(near[ix]); remaining-=len(ix)
            parts.append(np.concatenate(items)); near_parts.append(np.concatenate(near_items)); proposals.append(proposed)
        q=np.concatenate(parts); near=np.concatenate(near_parts)
        if len(np.unique(q,axis=0))!=sum(capacities): raise ValueError('Duplicate solid pool coordinates')
        allq.append(q); allnear.append(near)
        audit.append(dict(index=observed[i],proposed_by_region=proposals,pools=list(capacities),rings=len(rings)))
    q=np.stack(allq); near=np.stack(allnear); region=np.tile(np.repeat(np.arange(3,dtype=np.int8),capacities),(len(observed),1))
    meta=dict(schema='plane_auto_occ_area_uniform_v2',case_schema='plane_auto_occ_solid_v1',source_sha256=digest(case),
        observed=list(observed),seed=seed,generation=generation,capacities=list(capacities),margin_mm=margin_mm,
        boundary_band_mm=boundary_band_mm,near_mm=near_mm,contour_sha256=hashlib.sha256(allpoints.tobytes()).hexdigest(),
        hidden_geometry_used=False,region_names=['inside','outside_a','outside_b'],
        sampling='uniform filled components; uniform outside in observed-only ROI; exact band rejection',
        pool_sha256=hashlib.sha256(q.tobytes()+region.tobytes()+near.tobytes()).hexdigest(),planes=audit,
        build_seconds=time.perf_counter()-start)
    output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,q=q,region=region,near=near,occ=(region==0).astype(np.float32),metadata_json=np.array(json.dumps(meta)))
    output.with_suffix('.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta),flush=True)
    return meta
