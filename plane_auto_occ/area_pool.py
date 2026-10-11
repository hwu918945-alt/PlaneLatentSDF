"""Observed-only area-uniform class pools, exact boundary-band rejection."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
from matplotlib.path import Path as PolygonPath
from scipy.spatial import cKDTree
from shapely.geometry import Polygon
from .data import load_case
from .inference import bounds


def digest(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PlaneClassifier:
    def __init__(self, epi, endo):
        self.polys=[]; self.paths=[]; self.edges=[]
        for poly in (epi,endo):
            p=np.asarray(poly,np.float64)
            p=p[np.linalg.norm(p-np.roll(p,1,axis=0),axis=1)>1e-10]
            shape=Polygon(p)
            if not shape.is_valid or shape.area<=0: raise ValueError('Invalid polygon')
            self.polys.append(shape); self.paths.append(PolygonPath(np.vstack([p,p[0]]),closed=True))
            delta=np.roll(p,-1,axis=0)-p; mid=p+delta/2
            self.edges.append((p,delta,(delta*delta).sum(1),cKDTree(mid),np.linalg.norm(delta,axis=1).max()/2))
        if not self.polys[0].contains(self.polys[1]): raise ValueError('Endo must be strictly inside Epi')
        if self.polys[0].boundary.distance(self.polys[1].boundary)<=1e-8: raise ValueError('Intersecting boundaries')

    def band_distance(self, points, edge, radius=3.):
        # A segment within radius has its midpoint within radius+max_half_length.
        # Hence the KD-tree search cannot omit a segment relevant to either band.
        p,d,l2,tree,half=edge; reach=radius+half+1e-10
        nearest=tree.query(points)[0]; ids=np.flatnonzero(nearest<=reach)
        result=np.full(len(points),np.inf)
        if not len(ids): return result
        lists=tree.query_ball_point(points[ids],reach)
        lengths=np.array([len(x) for x in lists]); pid=np.repeat(ids,lengths)
        sid=np.concatenate(lists).astype(np.int64)
        diff=points[pid]-p[sid]; t=np.clip((diff*d[sid]).sum(1)/l2[sid],0,1)
        distances=np.linalg.norm(diff-t[:,None]*d[sid],axis=1)
        np.minimum.at(result,pid,distances)
        return result

    def classify(self, points, band=.05, near_mm=3.):
        inside_epi=self.paths[0].contains_points(points); inside_endo=self.paths[1].contains_points(points)
        region=np.where(inside_endo,1,np.where(inside_epi,0,2)).astype(np.int8)
        dist=np.minimum(*(self.band_distance(points,e,max(near_mm,band)) for e in self.edges))
        valid=dist>band
        if np.any(inside_endo & ~inside_epi & valid): raise ValueError('Invalid nested classification')
        return region,valid,dist<=near_mm


def build_pool(case, observed, output, capacities=(131072,65536,65536), seed=19, generation=0,
               margin_mm=20., boundary_band_mm=.05, near_mm=3., max_candidates=20000000):
    start=time.perf_counter(); out=Path(output)
    if out.exists(): raise FileExistsError(out)
    c=load_case(case,observed=observed)  # Select first; all following geometry is observed only.
    lower,upper,basis=bounds(c,margin_mm); scale=c['scale_mm']
    rng=np.random.default_rng(np.random.SeedSequence([seed,48191,generation]))
    all_q=[]; all_near=[]; audit=[]
    for i,pos in enumerate(c['s'].numpy()):
        classifier=PlaneClassifier(c['boundary'][i].numpy() @ basis[:,:2]*scale,
                                   c['boundary_endo'][i].numpy() @ basis[:,:2]*scale)
        remaining=list(capacities); parts=[[],[],[]]; nears=[[],[],[]]; proposed=0
        while any(remaining):
            size=65536; proposed+=size
            if proposed>max_candidates:
                raise ValueError(f'Plane {observed[i]} insufficient accepted pool: remaining {remaining}, candidates {proposed}')
            uv=rng.uniform(lower[:2],upper[:2],(size,2))
            q=(uv @ basis[:,:2].T+float(pos)*basis[:,2]).astype(np.float32)
            # Decide labels on the stored float32 coordinates, not pre-rounding candidates.
            region,valid,near=classifier.classify(q.astype(np.float64) @ basis[:,:2]*scale,boundary_band_mm,near_mm)
            for r in range(3):
                ids=np.flatnonzero(valid & (region==r))[:remaining[r]]
                if len(ids): parts[r].append(q[ids]); nears[r].append(near[ids]); remaining[r]-=len(ids)
        plane=np.concatenate([np.concatenate(v) for v in parts]); near=np.concatenate([np.concatenate(v) for v in nears])
        if len(np.unique(plane,axis=0))!=sum(capacities): raise ValueError(f'Plane {observed[i]} duplicate coordinates')
        if np.max(np.abs(plane @ c['axis'].numpy()-pos))>1e-5: raise ValueError('Axis-coordinate mismatch')
        all_q.append(plane); all_near.append(near)
        audit.append(dict(index=observed[i],proposed=proposed,pools=list(capacities),
            near_counts=[int(near[sum(capacities[:r]):sum(capacities[:r+1])].sum()) for r in range(3)]))
    q=np.stack(all_q); near=np.stack(all_near)
    region=np.tile(np.repeat(np.arange(3,dtype=np.int8),capacities),(len(observed),1))
    contour_hash=hashlib.sha256(c['boundary'].numpy().tobytes()+c['boundary_endo'].numpy().tobytes()).hexdigest()
    meta=dict(schema='plane_auto_occ_area_uniform_v2',source_sha256=digest(case),observed=list(observed),
        seed=seed,generation=generation,capacities=list(capacities),roi_stack_mm=(np.stack((lower,upper))*scale).tolist(),
        margin_mm=margin_mm,boundary_band_mm=boundary_band_mm,near_mm=near_mm,contour_sha256=contour_hash,
        hidden_geometry_used=False,pool_sha256=hashlib.sha256(q.tobytes()+region.tobytes()+near.tobytes()).hexdigest(),
        planes=audit,build_seconds=time.perf_counter()-start)
    out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,q=q,region=region,occ=(region==0).astype(np.float32),near=near,
        observed=np.array(observed),s=c['s'].numpy(),axis=c['axis'].numpy(),boundary=c['boundary'].numpy(),
        boundary_endo=c['boundary_endo'].numpy(),center_mm=c['center_mm'],scale_mm=np.array(scale),
        schema_version=np.array(meta['schema']),metadata_json=np.array(json.dumps(meta)))
    out.with_suffix('.json').write_text(json.dumps(meta,indent=2)); print(json.dumps(meta),flush=True)
    return meta


class AreaSampler:
    def __init__(self,path,seed=19):
        self.path=str(Path(path).resolve())
        with np.load(path,allow_pickle=False) as z:
            self.q=np.array(z['q']); self.near=np.array(z['near']); self.region=np.array(z['region'])
            self.meta=json.loads(str(z['metadata_json']))
        if self.meta['schema']!='plane_auto_occ_area_uniform_v2': raise ValueError('Wrong pool schema')
        if hashlib.sha256(self.q.tobytes()+self.region.tobytes()+self.near.tobytes()).hexdigest()!=self.meta['pool_sha256']:
            raise ValueError('Pool hash mismatch')
        self.capacities=self.meta['capacities']; self.rng=np.random.default_rng(np.random.SeedSequence([seed,99173,self.meta['generation']]))
        self.order={}; self.offset={}; self.cycles={}
        for i in range(len(self.q)):
            for r,cap in enumerate(self.capacities):
                key=(i,r); self.order[key]=self.rng.permutation(cap); self.offset[key]=0; self.cycles[key]=0
        self.last_stats={}

    def draw(self,i,count):
        if count%4: raise ValueError('Region budget must be divisible by four')
        budgets=[count//2,count//4,count//4]; ids=[]; stats=[]
        for r,n in enumerate(budgets):
            key=(i,r); cap=self.capacities[r]
            if n>cap or cap%n: raise ValueError('Pool capacity must be an integer multiple of budget')
            if self.offset[key]==cap:
                self.order[key]=self.rng.permutation(cap);self.offset[key]=0;self.cycles[key]+=1
            start=self.offset[key]; part=self.order[key][start:start+n]+sum(self.capacities[:r]); self.offset[key]+=n
            ids.append(part); near=int(self.near[i,part].sum())
            stats.append(dict(region=r,pool_size=cap,independent_points=n,duplicate_rate=0.,near=near,far=n-near,
                coverage_fraction=self.offset[key]/cap,completed_cycles=self.cycles[key]+int(self.offset[key]==cap)))
        index=np.concatenate(ids); self.last_stats[i]=stats
        return self.q[i,index].copy(),(self.region[i,index]==0).astype(np.float32)


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--output',required=True)
    p.add_argument('--observed',default='0,3,6');p.add_argument('--seed',type=int,default=19)
    a=p.parse_args();build_pool(a.case,[int(v) for v in a.observed.split(',')],a.output,seed=a.seed)


if __name__=='__main__':main()
