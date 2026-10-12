"""Observed-only physical-mm geometry adapter; immutable mmap pair-unit pools."""
import hashlib,json,math,shutil,uuid
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from .area_pool import digest
from .data import load_case
from .inference import bounds
from .paired_geometry import Plane

SCHEMA='paired_multiscale_roi_v1'


def contract(cfg):
    points=int(cfg['points'])
    if points%32:raise ValueError('Pair/ROI budget requires points divisible by 32')
    if int(cfg['pool_refresh_updates'])!=80:raise ValueError('This protocol preserves 80-update refresh')
    return dict(schema=SCHEMA,points=points,epi_pairs=points*3//32,endo_pairs=points*3//32,
        roi_points=points*5//8,pool_cycles=8,scales=list(cfg.get('pair_scales_mm',[.25,.75,2.])),
        scale_weights=list(cfg.get('pair_scale_weights',[.5,.3,.2])),
        boundary_band_mm=float(cfg['boundary_band_mm']),margin_mm=float(cfg['pool_margin_mm']),
        roi_reference_observed=list(cfg['roi_reference_observed']),seed=int(cfg['seed']),
        geometry_version='attached_e5567b511cd6_v1',cap_fraction=.4,jitter_fraction=.2)


def location(root,c,cfg,generation):
    cc=contract(cfg);identity=dict(source_sha256=c.get('_source_sha256') or digest(c['path']),observed=c['observed'],contract=cc)
    token=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
    return Path(root)/f"{c['case_id']}_{'ED' if c['phase']==0 else 'ES'}"/token/f'generation_{generation:04d}'


def check(path,c,cfg,generation,full=False):
    path=Path(path);m=json.loads((path/'metadata.json').read_text())
    for k,v in dict(contract=contract(cfg),source_sha256=c.get('_source_sha256') or digest(c['path']),observed=c['observed'],generation=int(generation)).items():
        if m[k]!=v:raise ValueError(f'Paired pool mismatch: {k}')
    for name,row in m['files'].items():
        f=path/name
        if f.stat().st_size!=row['bytes'] or (full and digest(f)!=row['sha256']):raise ValueError(f'Incomplete/changed pool {f}')
    return m


def _classifier(c,cfg,i,basis,lower,upper):
    scale=c['scale_mm'];epi=c['boundary'][i].numpy()@basis[:,:2]*scale;endo=c['boundary_endo'][i].numpy()@basis[:,:2]*scale
    return Plane(epi,endo,lower[:2]*scale,upper[:2]*scale,float(cfg['boundary_band_mm']))


def build(case,observed,root,cfg,generation):
    c=load_case(case,observed=observed);target=location(root,c,cfg,generation)
    if target.exists():return check(target,c,cfg,generation,full=True)
    cc=contract(cfg);refs=cc['roi_reference_observed']
    if not set(refs)<=set(c['observed']):raise ValueError('ROI reference must be a subset of permitted observations')
    ref=load_case(case,observed=refs);lower,upper,basis=bounds(ref,cc['margin_mm']);scale=c['scale_mm']
    identity=int(hashlib.sha256(f"{c['case_id']}:{c['phase']}".encode()).hexdigest()[:8],16)
    rng=np.random.default_rng(np.random.SeedSequence([cfg['seed'],identity,48191,generation]))
    epcap=cc['epi_pairs']*8;encap=cc['endo_pairs']*8;ucap=cc['roi_points']*8;size=cc['points']*8
    shape=(len(c['s']),size)
    arrays=dict(q=np.empty((*shape,3),np.float32),region=np.empty(shape,np.int8),source=np.empty(shape,np.int8),
        scale_id=np.empty(shape,np.int8),delta_mm=np.empty(shape,np.float32),pair_id=np.empty(shape,np.int32))
    audit=[]
    for i,pos in enumerate(c['s'].numpy()):
        plane=_classifier(c,cfg,i,basis,lower,upper);uvs=[];sources=[];scales=[];deltas=[];pairids=[];pa=[]
        for b,count in [(0,epcap),(1,encap)]:
            bank=plane.pairs(b,count,rng,scales=cc['scales'],weights=cc['scale_weights'])
            uvs.append(bank['q'].reshape(-1,2));sources.append(np.tile([0,1] if b==0 else [2,3],count).astype(np.int8))
            scales.append(np.repeat(bank['scale_id'],2).astype(np.int8));deltas.append(np.repeat(bank['delta'],2).astype(np.float32))
            pairids.append(np.repeat(np.arange(count,dtype=np.int32),2))
            pa.append(dict(boundary=b,pairs=count,capped=int(bank['capped'].sum()),rotated=int(bank['rotated'].sum()),
                initial_invalid=int((~bank['initial_good']).sum()),minimum_delta_mm=float(bank['delta'].min())))
        uvs.append(plane.uniform(ucap,rng));sources.append(np.full(ucap,4,np.int8));scales.append(np.full(ucap,-1,np.int8))
        deltas.append(np.zeros(ucap,np.float32));pairids.append(np.full(ucap,-1,np.int32))
        uv=np.concatenate(uvs);q=(uv/scale@basis[:,:2].T+float(pos)*basis[:,2]).astype(np.float32)
        # Recheck final normalized float32 WORLD coordinates, after mapping back to mm.
        region,valid,_=plane.classify(q.astype(np.float64)@basis[:,:2]*scale)
        source=np.concatenate(sources);expected=np.array([0,2,0,1],np.int8)
        if not valid.all() or not np.all(region[source<4]==expected[source[source<4]]):
            raise ValueError(f'Final coordinate mapping invalid: {c["case_id"]} phase{c["phase"]} plane{i}')
        if len(np.unique(q,axis=0))!=size:raise ValueError('Duplicate final float32 point: no silent replacement')
        if np.max(abs(q@c['axis'].numpy()-pos))>1e-5:raise ValueError('Point left its observed plane')
        arrays['q'][i]=q;arrays['region'][i]=region;arrays['source'][i]=source
        arrays['scale_id'][i]=np.concatenate(scales);arrays['delta_mm'][i]=np.concatenate(deltas);arrays['pair_id'][i]=np.concatenate(pairids)
        audit.append(dict(index=c['observed'][i],pair_geometry=pa,roi_positive_points=int((region[source==4]==0).sum())))
    target.parent.mkdir(parents=True,exist_ok=True);tmp=target.parent/(target.name+'.partial-'+uuid.uuid4().hex);tmp.mkdir()
    m=dict(schema=SCHEMA,contract=cc,source_sha256=digest(case),observed=c['observed'],generation=int(generation),
        case_id=c['case_id'],phase=c['phase'],roi_stack_mm=(np.stack([lower,upper])*scale).tolist(),
        hidden_geometry_used=False,source_enum={'0':'epi_myo','1':'epi_background','2':'endo_myo','3':'endo_cavity','4':'roi_uniform'},
        planes=audit,files={})
    try:
        for name,a in arrays.items():
            f=tmp/(name+'.npy');np.save(f,a);m['files'][f.name]=dict(bytes=f.stat().st_size,sha256=digest(f))
        (tmp/'metadata.json').write_text(json.dumps(m,indent=2));tmp.rename(target)
    except BaseException:shutil.rmtree(tmp,ignore_errors=True);raise
    return m


class PairedSampler:
    def __init__(self,path,metadata):
        self.path=str(path);self.meta=metadata;self.contract=metadata['contract'];self.orders={}
        self.q=np.load(Path(path)/'q.npy',mmap_mode='r');self.region=np.load(Path(path)/'region.npy',mmap_mode='r')
        self.source=np.load(Path(path)/'source.npy',mmap_mode='r');self.last_stats={};self.last_source={}
    def set_visit(self,visit):self.visit=int(visit)
    def draw(self,i,count):
        cc=self.contract
        if count!=cc['points']:raise ValueError('Changed point budget')
        cycle,offset=divmod(self.visit,8);indices=[];start=0
        for key,budget,pair in [(0,cc['epi_pairs'],True),(1,cc['endo_pairs'],True),(2,cc['roi_points'],False)]:
            cap=8*budget;cachekey=(i,key,cycle)
            if cachekey not in self.orders:
                self.orders={k:v for k,v in self.orders.items() if k[:2]!=(i,key)}
                self.orders[cachekey]=np.random.default_rng(np.random.SeedSequence([cc['seed'],99173,self.meta['generation'],i,key,cycle])).permutation(cap)
            unit=self.orders[cachekey][offset*budget:(offset+1)*budget]
            ids=(unit[:,None]*2+np.array([0,1])).reshape(-1) if pair else unit
            indices.append(ids+start);start+=cap*(2 if pair else 1)
        ids=np.concatenate(indices);q=self.q[i,ids].copy();source=self.source[i,ids].copy();y=(self.region[i,ids]==0).astype(np.float32)
        self.last_source[i]=source
        self.last_stats[i]=dict(independent_points=count,source_counts={str(k):int((source==k).sum()) for k in range(5)},
            paired_unit_preserved=True,coverage_cycle=cycle,cycle_fraction=(offset+1)/8,
            generation=self.meta['generation'],roi_positive_fraction=float(y[source==4].mean()))
        return q,y


class StaticPairedSampler:
    def __init__(self,sampler,points,planes):
        self.meta=sampler.meta;self.arrays=[];self.last_stats={};self.last_source={}
        for i in range(planes):
            self.arrays.append(sampler.draw(i,points));self.last_stats[i]=sampler.last_stats[i];self.last_source[i]=sampler.last_source[i]
    def draw(self,i,count):
        q,y=self.arrays[i]
        if len(q)!=count:raise ValueError('Changed static point budget')
        return q,y


class PairedPoolStore:
    """Same get/prefetch/close interface as upstream OfflinePoolStore."""
    def __init__(self,root,cfg,max_loaded=32):self.root=root;self.cfg=cfg;self.cache={}
    def prefetch(self,requests):pass  # Read-only mmap; all geometry completed before GPU fitting.
    def get(self,c,visits):
        generation=visits//self.cfg['pool_refresh_updates'];path=location(self.root,c,self.cfg,generation)
        if not path.exists():raise FileNotFoundError(f'Paired pool missing: {path}; run suite precache first')
        key=str(path)
        if key not in self.cache:self.cache={key:PairedSampler(path,check(path,c,self.cfg,generation,full=True))}
        sampler=self.cache[key];sampler.set_visit(visits%self.cfg['pool_refresh_updates'])
        return StaticPairedSampler(sampler,self.cfg['points'],len(c['s']))
    def close(self):self.cache.clear()
