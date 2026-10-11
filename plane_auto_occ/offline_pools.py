"""Atomic, immutable, memory-mapped point pools prepared outside training."""
import hashlib,json,shutil,time,uuid
from pathlib import Path
import numpy as np
from .area_pool import digest
from .data import load_case
from .inference import bounds
from .fast_geometry import CompiledPlaneClassifier


def identity(case_id,phase):
    return hashlib.sha256(f'{case_id}:{phase}'.encode()).hexdigest()[:16]


def directory(root,c,generation):
    observations=hashlib.sha256(json.dumps(c['observed']).encode()).hexdigest()[:10]
    return Path(root)/identity(c['case_id'],c['phase'])/observations/f'generation_{generation:04d}'


def validate(path,c,cfg,generation,full=False):
    path=Path(path);m=json.loads((path/'metadata.json').read_text())
    expected=dict(source_sha256=c['_source_sha256'] if '_source_sha256' in c else digest(c['path']),observed=c['observed'],generation=generation,
        capacities=list(cfg['pool_capacities']),seed=(cfg['seed']+int(identity(c['case_id'],c['phase'])[:8],16))%(2**32),
        margin_mm=cfg['pool_margin_mm'],boundary_band_mm=cfg['boundary_band_mm'],near_mm=cfg['near_mm'])
    for k,v in expected.items():
        if m[k]!=v:raise ValueError(f'Offline pool contract mismatch {k}: {path}')
    for name,row in m['files'].items():
        if (path/name).stat().st_size!=row['bytes']:raise ValueError(f'Incomplete offline array: {path/name}')
        if full and digest(path/name)!=row['sha256']:raise ValueError(f'Offline array changed: {path/name}')
    return m


def publish(target,q,near,metadata):
    target=Path(target);target.parent.mkdir(parents=True,exist_ok=True)
    temporary=target.parent/(target.name+'.partial-'+uuid.uuid4().hex)
    temporary.mkdir()
    try:
        np.save(temporary/'q.npy',q)
        np.save(temporary/'near_bits.npy',np.packbits(near,axis=1,bitorder='little'))
        # Regional labels are implicit in fixed contiguous 1/2,1/4,1/4 pools.
        region=np.tile(np.repeat(np.arange(3,dtype=np.int8),metadata['capacities']),(len(q),1))
        metadata=dict(metadata,pool_sha256=hashlib.sha256(q.tobytes()+region.tobytes()+near.tobytes()).hexdigest(),
            storage='float32 world xyz + lossless bitpacked near; implicit class labels',files={})
        for name in ('q.npy','near_bits.npy'):
            metadata['files'][name]=dict(bytes=(temporary/name).stat().st_size,sha256=digest(temporary/name))
        (temporary/'metadata.json').write_text(json.dumps(metadata,indent=2))
        temporary.rename(target)
    except BaseException:
        shutil.rmtree(temporary,ignore_errors=True);raise
    return metadata


def prepare(case,observed,root,cfg,generation,legacy_root=None,backend='compiled_cpu'):
    start=time.perf_counter();c=load_case(case,observed=observed);target=directory(root,c,generation)
    if target.exists():return validate(target,c,cfg,generation,full=True)
    seed=(cfg['seed']+int(identity(c['case_id'],c['phase'])[:8],16))%(2**32)
    base=dict(schema='plane_auto_occ_offline_pool_v1',source_sha256=digest(case),observed=c['observed'],
        seed=seed,generation=generation,capacities=list(cfg['pool_capacities']),margin_mm=cfg['pool_margin_mm'],
        boundary_band_mm=cfg['boundary_band_mm'],near_mm=cfg['near_mm'],hidden_geometry_used=False,
        backend=backend,case_id=c['case_id'],phase=c['phase'])
    legacy=Path(legacy_root)/identity(c['case_id'],c['phase'])/f'generation_{generation:04d}.npz' if legacy_root else None
    if legacy and legacy.exists():
        with np.load(legacy,allow_pickle=False) as z:
            prior=json.loads(str(z['metadata_json']));q=np.array(z['q']);near=np.array(z['near']);region=np.array(z['region'])
        for k in ('source_sha256','observed','seed','generation','capacities','margin_mm','boundary_band_mm','near_mm'):
            if prior[k]!=base[k]:raise ValueError(f'Legacy pool contract mismatch {k}')
        assert hashlib.sha256(q.tobytes()+region.tobytes()+near.tobytes()).hexdigest()==prior['pool_sha256']
        assert np.array_equal(region,np.tile(np.repeat(np.arange(3,dtype=np.int8),cfg['pool_capacities']),(len(q),1)))
        base.update(backend='exact legacy import',legacy_sha256=digest(legacy),duplicates_replaced=0,build_seconds=time.perf_counter()-start)
        return publish(target,q,near,base)
    lower,upper,basis=bounds(c,cfg['pool_margin_mm']);scale=c['scale_mm']
    rng=np.random.default_rng(np.random.SeedSequence([seed,48191,generation]));allq=[];allnear=[];audit=[];removed=0
    for i,pos in enumerate(c['s'].numpy()):
        epi=c['boundary'][i].numpy()@basis[:,:2]*scale;endo=c['boundary_endo'][i].numpy()@basis[:,:2]*scale
        if backend=='cuda':
            from .cuda_geometry import CudaPlaneClassifier
            classifier=CudaPlaneClassifier(epi,endo)
        elif backend=='reference':
            from .area_pool import PlaneClassifier
            classifier=PlaneClassifier(epi,endo)
        elif backend=='compiled_cpu':classifier=CompiledPlaneClassifier(epi,endo)
        else:raise ValueError('Unknown geometry backend')
        boxes=[(epi.min(0)/scale,epi.max(0)/scale),(endo.min(0)/scale,endo.max(0)/scale),(lower[:2],upper[:2])]
        parts=[];nears=[];proposals=[]
        for r,cap in enumerate(cfg['pool_capacities']):
            items=[];nearitems=[];remaining=cap;proposed=0
            while True:
                while remaining:
                    size=65536;proposed+=size
                    if proposed>20000000:raise ValueError(f'Insufficient valid area: {c["case_id"]} plane{i} region{r}')
                    uv=rng.uniform(*boxes[r],(size,2));q=(uv@basis[:,:2].T+float(pos)*basis[:,2]).astype(np.float32)
                    region,valid,near=classifier.classify(q.astype(np.float64)@basis[:,:2]*scale,cfg['boundary_band_mm'],cfg['near_mm'])
                    ix=np.flatnonzero(valid&(region==r))[:remaining]
                    if len(ix):items.append(q[ix]);nearitems.append(near[ix]);remaining-=len(ix)
                q=np.concatenate(items);near=np.concatenate(nearitems)
                _,indices=np.unique(q,axis=0,return_index=True);indices.sort()
                if len(indices)==cap:break
                removed+=len(q)-len(indices);items=[q[indices]];nearitems=[near[indices]];remaining=cap-len(indices)
            parts.append(q);nears.append(near);proposals.append(proposed)
        q=np.concatenate(parts);near=np.concatenate(nears)
        if len(np.unique(q,axis=0))!=sum(cfg['pool_capacities']):raise ValueError('Cross-region duplicate')
        if np.max(abs(q@c['axis'].numpy()-pos))>1e-5:raise ValueError('Plane mismatch')
        allq.append(q);allnear.append(near);audit.append(dict(index=c['observed'][i],proposed_by_region=proposals))
    base.update(duplicates_replaced=removed,planes=audit,build_seconds=time.perf_counter()-start,
        conditional_area_density='original class-containing proposal boxes and uniform true-region acceptance')
    return publish(target,np.stack(allq),np.stack(allnear),base)


class OfflineSampler:
    """Exact VisitSampler permutations; mmap and lazy orders, no ZIP or pool rebuild."""
    def __init__(self,path,metadata=None):
        self.path=str(path);self.meta=metadata if metadata is not None else json.loads((Path(path)/'metadata.json').read_text());self.capacities=self.meta['capacities']
        self.q=np.load(Path(path)/'q.npy',mmap_mode='r');self.near_bits=np.load(Path(path)/'near_bits.npy',mmap_mode='r')
        self.orders={};self.last_stats={}

    def set_visit(self,visit,points,seed):self.visit=int(visit);self.points=int(points);self.seed=int(seed)

    def draw(self,i,count):
        if count!=self.points or count%4:raise ValueError('Changed point budget')
        budgets=[count//2,count//4,count//4];ids=[];stats=[]
        for r,n in enumerate(budgets):
            cap=self.capacities[r]
            if cap%n:raise ValueError('Invalid pool capacity')
            draws=cap//n;cycle=self.visit//draws;offset=self.visit%draws*n;key=(i,r,cycle)
            if key not in self.orders:
                self.orders[key]=np.random.default_rng(np.random.SeedSequence([self.seed,99173,self.meta['generation'],i,r,cycle])).permutation(cap)
                for old in list(self.orders):
                    if old[:2]==key[:2] and old!=key:del self.orders[old]
            ix=self.orders[key][offset:offset+n]+sum(self.capacities[:r]);ids.append(ix)
            near=int(((self.near_bits[i,ix//8]>>(ix%8))&1).sum())
            stats.append(dict(region=r,pool_size=cap,independent_points=n,duplicate_rate=0.,near=near,far=n-near,
                coverage_fraction=(offset+n)/cap,completed_cycles=cycle+int(offset+n==cap)))
        ix=np.concatenate(ids);self.last_stats[i]=stats
        return self.q[i,ix].copy(),np.repeat(np.array([1.,0.,0.],np.float32),budgets)
