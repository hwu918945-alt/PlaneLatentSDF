"""Mathematical equivalence, sampling provenance, resume and frozen-fit contracts."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from .repair_tests import fixture
from .data import load_case
from .model import AutoOccupancy
from .chunked import backward_case as reference
from .fast_chunked import backward_case as fast
from .offline_pools import prepare,directory,OfflineSampler
from .offline_store import StaticSampler
from .population import VisitSampler,fit
from .area_pool import PlaneClassifier,digest
from .fast_geometry import CompiledPlaneClassifier

class Fixed:
    def __init__(self,n,points):
        rng=np.random.default_rng(49);self.arrays=[(rng.uniform(-.25,.25,(points,3)).astype(np.float32),np.r_[np.ones(points//2,np.float32),np.zeros(points//2,np.float32)]) for _ in range(n)];self.last_stats={i:[] for i in range(n)}
    def draw(self,i,count):return self.arrays[i]

class FastContracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);torch.set_num_threads(1)
        self.cfg=json.loads(Path(__file__).with_name('config.json').read_text());self.cfg.update(width=32,points=128,microbatch_points=32,fast_query_points=197,pool_capacities=[512,256,256],pool_refresh_updates=2,save_every=1,log_every=100,offline_pool_root=str(self.root/'pools'),pipeline_backend='offline_batched_v1')
    def tearDown(self):self.tmp.cleanup()
    def case(self,name):
        p=self.root/f'{name}.npz';fixture(p)
        with np.load(p) as z:r={k:np.array(z[k]) for k in z.files}
        r['case_id']=np.array(name);np.savez_compressed(p,**r);return str(p)
    def prepare(self,path):
        for g in range(3):prepare(path,[0,3,6],self.root/'pools',self.cfg,g)
    def test_completecase_loss_gradients_and_single_adam(self):
        for n in (3,7,12):
            for beta in (0.,.25):
                torch.manual_seed(1);m=AutoOccupancy(128,32);z=torch.nn.Parameter(torch.randn(128)*.1)
                b=copy.deepcopy(m);w=torch.nn.Parameter(z.detach().clone());cfg=dict(self.cfg,loo_mix=beta)
                c=dict(s=torch.linspace(-.2,.2,n),axis=torch.tensor([0.,0.,1.]),phase=0,observed=list(range(n)),_area_sampler=Fixed(n,128))
                a,sa=reference(m,z,c,cfg,0);d,sd=fast(b,w,c,cfg,0)
                torch.testing.assert_close(a,d,rtol=2e-6,atol=2e-7)
                for p,q in zip([z]+list(m.parameters()),[w]+list(b.parameters())):torch.testing.assert_close(p.grad,q.grad,rtol=1e-4,atol=2e-7)
                self.assertEqual(sa['independent_points'],sd['independent_points']);self.assertEqual(sa['decoder_queries'],sd['decoder_queries'])
                oa=torch.optim.Adam([z]+list(m.parameters()),lr=.001);ob=torch.optim.Adam([w]+list(b.parameters()),lr=.001)
                oa.step();ob.step()
                for p,q in zip([z]+list(m.parameters()),[w]+list(b.parameters())):torch.testing.assert_close(p,q,rtol=2e-5,atol=2e-6)
    def test_indexed_classifier_labels_and_bands(self):
        t=np.linspace(0,2*np.pi,256,endpoint=False);epi=np.c_[20*np.cos(t),15*np.sin(t)];endo=epi*.7
        a=PlaneClassifier(epi,endo);b=CompiledPlaneClassifier(epi,endo)
        rng=np.random.default_rng(12);p=rng.uniform(-30,30,(60000,2))
        normals=np.c_[np.cos(t),np.sin(t)];p=np.concatenate([p]+[epi+normals*d for d in (.049999,.050001,2.999999,3.000001,-.049999,-.050001)])
        for u,v in zip(a.classify(p),b.classify(p)):np.testing.assert_array_equal(u,v)
    def test_offline_sampler_coverage_and_resume(self):
        path=self.case('one');self.prepare(path);c=load_case(path,observed=[0,3,6]);p=directory(self.root/'pools',c,0);s=OfflineSampler(p);allpoints=[]
        for v in range(8):
            s.set_visit(v,128,19);q,y=s.draw(0,128);allpoints.append(q)
            b=OfflineSampler(p);b.set_visit(v,128,19);q2,y2=b.draw(0,128);np.testing.assert_array_equal(q,q2);self.assertEqual(s.last_stats,b.last_stats)
        self.assertEqual(len(np.unique(np.concatenate(allpoints),axis=0)),1024)
    def test_optimized_resume_and_frozen_weights(self):
        paths=[self.case('one'),self.case('two')]
        for p in paths:self.prepare(p)
        cfg=self.cfg;obs=[[0,3,6]]*2
        a=fit(paths,obs,cfg,self.root/'full',steps=6)
        calls=[0]
        def stop():calls[0]+=1;return calls[0]>3
        with self.assertRaises(InterruptedError):fit(paths,obs,cfg,self.root/'resume',steps=6,stop_requested=stop)
        b=fit(paths,obs,cfg,self.root/'resume',steps=6,resume=True)
        ca=torch.load(a,weights_only=False);cb=torch.load(b,weights_only=False)
        self.assertEqual(ca['pending_order'],cb['pending_order']);self.assertEqual(ca['visits'],cb['visits']);self.assertTrue(torch.equal(ca['codes'],cb['codes']))
        for k,v in ca['model'].items():self.assertTrue(torch.equal(v,cb['model'][k]),k)
        for k,st in ca['optimizer']['state'].items():
            for name,v in st.items():self.assertTrue(torch.equal(v,cb['optimizer']['state'][k][name]))
        p=self.case('heldout');self.prepare(p);f=fit([p],[[0,3,6]],cfg,self.root/'heldout',steps=2,pretrained=a);cf=torch.load(f,weights_only=False)
        for k,v in ca['model'].items():self.assertTrue(torch.equal(v,cf['model'][k]))
    def test_missing_cache_checkpoints_last_successful_update(self):
        from unittest.mock import patch
        from .offline_store import OfflinePoolStore
        paths=[self.case('one'),self.case('two')]
        for p in paths:self.prepare(p)
        cfg=dict(self.cfg,save_every=250);observed=[[0,3,6]]*2
        full=fit(paths,observed,cfg,self.root/'full',steps=4);original=OfflinePoolStore.get;calls=[0]
        def fail_on_second(store,c,visits):
            calls[0]+=1
            if calls[0]==2:raise FileNotFoundError('Injected missing cache')
            return original(store,c,visits)
        with patch.object(OfflinePoolStore,'get',fail_on_second):
            with self.assertRaises(FileNotFoundError):fit(paths,observed,cfg,self.root/'failure',steps=4)
        checkpoint=torch.load(self.root/'failure/last.pt',weights_only=False);self.assertEqual(checkpoint['step'],1)
        resumed=fit(paths,observed,cfg,self.root/'failure',steps=4,resume=True)
        a=torch.load(full,weights_only=False);b=torch.load(resumed,weights_only=False)
        self.assertTrue(torch.equal(a['codes'],b['codes']));self.assertEqual(a['visits'],b['visits'])
        for k,v in a['model'].items():self.assertTrue(torch.equal(v,b['model'][k]))
    def test_hidden_contours_do_not_change_offline_pool(self):
        base=self.root/'base.npz';hidden=self.root/'hidden.npz';fixture(base);fixture(hidden,hidden_shift=.5)
        a=prepare(base,[0,3,6],self.root/'a',self.cfg,0);b=prepare(hidden,[0,3,6],self.root/'b',self.cfg,0)
        self.assertEqual(a['pool_sha256'],b['pool_sha256'])
    def test_full_pipeline_validation_selection_and_cache_retirement(self):
        from .run_experiment import heart
        root=self.root/'experiment';root.mkdir();cfg=dict(self.cfg,offline_pool_root=str(root/'point_pools'))
        lock={'splits':{}}
        for split in ('train','val','test'):
            rows=[]
            for phase in (0,1):
                path=root/'data'/split/f'{split}_{phase}.npz';path.parent.mkdir(parents=True,exist_ok=True);fixture(path)
                with np.load(path) as z:r={k:np.array(z[k]) for k in z.files}
                r['case_id']=np.array(split);r['phase']=np.array(phase);np.savez_compressed(path,**r)
                rows.append(dict(path=str(path.relative_to(root)),case_id=split,phase=phase,planes=7,sha256=digest(path)))
                prepare(path,None if split=='train' else [0,3,6],cfg['offline_pool_root'],cfg,0)
            lock['splits'][split]={'records':rows}
        plan=dict(train_epochs=2,total_updates=4,val_epochs=[1,2],visual_epochs=[0],inference_steps=2,resolution=16,mesh_resolution=8)
        for name,value in [('plan.json',plan),('config.json',cfg),('data_lock.json',lock)]: (root/name).write_text(json.dumps(value))
        (root/'logs').mkdir();heart(root,plan,cfg,lock,'cpu')
        selection=json.loads((root/'outputs/CHECKPOINT_SELECTION.json').read_text())
        self.assertEqual(selection['selection_split'],'val');self.assertEqual([r['epoch'] for r in selection['candidates']],[1,2])
        self.assertTrue((root/'outputs/RUN_COMPLETE.json').exists());self.assertTrue((root/'cache_manifests/PRUNED_train_0000.json').exists())
        for row in lock['splits']['train']['records']:
            c=load_case(root/row['path']);p=directory(cfg['offline_pool_root'],c,0)
            self.assertFalse(p.exists());self.assertTrue(p.with_suffix('.retired.json').exists())
            # A saved older checkpoint can regenerate the exact retired pool.
            retired=json.loads(p.with_suffix('.retired.json').read_text());regenerated=prepare(root/row['path'],None,cfg['offline_pool_root'],cfg,0)
            self.assertEqual(retired['pool_sha256'],regenerated['pool_sha256'])
if __name__=='__main__':unittest.main()
