"""Read-only pool store and CPU prefetch, never geometry generation in an update."""
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from .offline_pools import directory,validate,OfflineSampler

class StaticSampler:
    def __init__(self,sampler,points):
        self.meta=sampler.meta;self.arrays=[];self.last_stats={}
        for i in range(len(sampler.q)):
            self.arrays.append(sampler.draw(i,points));self.last_stats[i]=sampler.last_stats[i]
    def draw(self,i,count):
        q,y=self.arrays[i]
        if len(q)!=count:raise ValueError('Point budget changed')
        return q,y

class OfflinePoolStore:
    def __init__(self,root,cfg,max_loaded=32):
        self.root=root;self.cfg=cfg;self.cache=OrderedDict();self.max_loaded=max_loaded
        self.validated={}
        self.executor=ThreadPoolExecutor(max_workers=1);self.pending={}
    def _get(self,c,visits):
        generation=visits//self.cfg['pool_refresh_updates'];path=directory(self.root,c,generation);key=str(path)
        if key in self.cache:sampler=self.cache.pop(key)
        else:
            if not (path/'q.npy').exists():raise FileNotFoundError(f'Offline pool missing: {path}. Run precache before training.')
            if key not in self.validated:self.validated[key]=validate(path,c,self.cfg,generation)
            sampler=OfflineSampler(path,self.validated[key])
        self.cache[key]=sampler
        while len(self.cache)>self.max_loaded:self.cache.popitem(last=False)
        sampler.set_visit(visits%self.cfg['pool_refresh_updates'],self.cfg['points'],self.cfg['seed'])
        return StaticSampler(sampler,self.cfg['points'])
    def prefetch(self,requests):
        for c,visits in requests:
            key=(c['path'],visits)
            if key not in self.pending:self.pending[key]=self.executor.submit(self._get,c,visits)
    def get(self,c,visits):
        key=(c['path'],visits)
        if key not in self.pending:self.prefetch([(c,visits)])
        return self.pending.pop(key).result()
    def close(self):
        self.executor.shutdown(wait=True);self.cache.clear();self.pending.clear()
