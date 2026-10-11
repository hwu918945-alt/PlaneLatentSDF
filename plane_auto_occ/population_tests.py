"""Population isolation/resume and disconnected/nested solid geometry contracts."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from .model import AutoOccupancy
from .population import fit,independent_codes,VisitSampler
from .solid import SolidClassifier,convert,load_case,build_pool
from .repair_tests import fixture


class PopulationContracts(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2);self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name)
    def tearDown(self):self.tmp.cleanup()
    def case(self,name,phase=0):
        p=self.root/f'{name}.npz';fixture(p)
        with np.load(p) as z:r={k:np.array(z[k]) for k in z.files}
        r['case_id']=np.array(name);r['phase']=np.array(phase);np.savez_compressed(p,**r)
        return str(p)
    def cfg(self):
        cfg=json.loads(Path(__file__).with_name('config.json').read_text())
        cfg.update(width=32,points=128,microbatch_points=32,pool_capacities=[512,256,256],save_every=1,
            pool_refresh_updates=2,log_every=100)
        return cfg

    def test_inactive_adam_code_and_state_remain_bit_identical(self):
        torch.manual_seed(5);codes=independent_codes(2,4,'cpu');opt=torch.optim.Adam(codes.parameters(),lr=.01)
        for i in (0,1):
            opt.zero_grad(set_to_none=True);codes[i].square().mean().backward();opt.step()
        saved=codes[0].detach().clone();state=copy.deepcopy(opt.state[codes[0]])
        opt.zero_grad(set_to_none=True);codes[1].square().mean().backward();opt.step()
        self.assertIsNone(codes[0].grad);self.assertTrue(torch.equal(saved,codes[0]))
        for k,v in state.items():self.assertTrue(torch.equal(v,opt.state[codes[0]][k]))

    def test_interrupted_resume_is_identical_with_case_pool_refresh(self):
        paths=[self.case('one'),self.case('two')];cfg=self.cfg();observed=[[0,3,6]]*2
        uninterrupted=fit(paths,observed,cfg,self.root/'full',steps=6)
        calls=[0]
        def stop():calls[0]+=1;return calls[0]>3
        with self.assertRaises(InterruptedError):fit(paths,observed,cfg,self.root/'resume',steps=6,stop_requested=stop)
        resumed=fit(paths,observed,cfg,self.root/'resume',steps=6,resume=True)
        a=torch.load(uninterrupted,weights_only=False);b=torch.load(resumed,weights_only=False)
        self.assertEqual(a['visits'],[3,3]);self.assertEqual(a['visits'],b['visits'])
        self.assertEqual(a['pending_order'],b['pending_order']);self.assertTrue(torch.equal(a['codes'],b['codes']))
        for k in a['model']:self.assertTrue(torch.equal(a['model'][k],b['model'][k]),k)
        for key,state in a['optimizer']['state'].items():
            for k,v in state.items():self.assertTrue(torch.equal(v,b['optimizer']['state'][key][k]))
        self.assertTrue(torch.equal(a['torch_rng'],b['torch_rng']))

    def test_diagnostics_rng_does_not_change_training(self):
        paths=[self.case('one'),self.case('two')];cfg=self.cfg()
        a=fit(paths,[[0,3,6]]*2,cfg,self.root/'without',steps=3)
        def noisy(*args):torch.manual_seed(999);np.random.seed(999);torch.rand(100)
        b=fit(paths,[[0,3,6]]*2,cfg,self.root/'with',steps=3,milestones=(0,1,2,3),diagnose=noisy)
        ca=torch.load(a,weights_only=False);cb=torch.load(b,weights_only=False)
        self.assertTrue(torch.equal(ca['codes'],cb['codes']))
        for k,v in ca['model'].items():self.assertTrue(torch.equal(v,cb['model'][k]))

    def test_frozen_inference_and_patient_leakage(self):
        cfg=self.cfg();a=self.case('train');b=self.case('heldout');es=self.case('train',1)
        prior=fit([a],[[0,3,6]],cfg,self.root/'pretrain',steps=1)
        result=fit([b],[[0,3,6]],cfg,self.root/'heldout',steps=2,pretrained=prior)
        ca=torch.load(prior,weights_only=False);cb=torch.load(result,weights_only=False)
        for k,v in ca['model'].items():self.assertTrue(torch.equal(v,cb['model'][k]))
        self.assertNotEqual(ca['records'][0]['case_id'],cb['records'][0]['case_id'])
        with self.assertRaises(ValueError):fit([es],[[0,3,6]],cfg,self.root/'leak',steps=1,pretrained=prior)

    def test_solid_islands_holes_and_exact_boundary_band(self):
        def square(x,y,size):return np.array([[x,y],[x+size,y],[x+size,y+size],[x,y+size]],float)
        c=SolidClassifier([square(0,0,4),square(1,1,2),square(8,0,2)])
        q=np.array([[.5,.5],[2,2],[9,1],[6,1],[.02,.5]])
        r,valid,near=c.classify(q)
        np.testing.assert_array_equal(r,[0,1,0,1,0]);np.testing.assert_array_equal(valid,[1,1,1,1,0])
        # After rejection the disconnected component probabilities are actual area,
        # not ring count, bounding-box overlap or artificial connecting polygons.
        uv,own=c.positive_proposals(np.random.default_rng(9),50000);r,v,_=c.classify(uv,band=0)
        accepted=uv[own & v & (r==0)];ratio=np.mean(accepted[:,0]>7)
        self.assertAlmostEqual(ratio,4/(12+4),delta=.015)

    def test_full_split_pipeline_selects_validation_then_runs_test(self):
        from .run_experiment import heart
        root=self.root/'experiment';root.mkdir();cfg=self.cfg();cfg['save_every']=2
        lock={'splits':{}}
        for split in ('train','val','test'):
            rows=[]
            for phase in (0,1):
                path=root/'data'/split/f'{split}_{phase}.npz';path.parent.mkdir(parents=True,exist_ok=True);fixture(path)
                with np.load(path) as z:r={k:np.array(z[k]) for k in z.files}
                r['case_id']=np.array(split);r['phase']=np.array(phase);np.savez_compressed(path,**r)
                rows.append(dict(path=str(path.relative_to(root)),case_id=split,phase=phase,planes=7))
            lock['splits'][split]={'records':rows}
        plan=dict(train_epochs=2,total_updates=4,val_epochs=[1,2],visual_epochs=[0],inference_steps=2,resolution=16,mesh_resolution=8)
        heart(root,plan,cfg,lock,'cpu')
        select=json.loads((root/'outputs/CHECKPOINT_SELECTION.json').read_text())
        self.assertEqual([r['epoch'] for r in select['candidates']],[1,2]);self.assertEqual(select['selection_split'],'val')
        best=max(select['candidates'],key=lambda x:(x['score'],-x['epoch']))
        self.assertEqual(select['best'],best)
        test=json.loads((root/'outputs/final_selected/test/SUMMARY.json').read_text())
        self.assertTrue(test['shared_frozen']);self.assertEqual(test['phase_records'],2);self.assertEqual(test['patients'],1)
        for row in test['records']:
            ck=torch.load(Path(row['metrics']).parents[1]/'fit/last.pt',weights_only=False)
            self.assertEqual(ck['records'][0]['observed'],[0,3,6]);self.assertEqual(ck['step'],2)
            self.assertEqual(ck['training_keys'],[['train',0],['train',1]])
        self.assertTrue((root/'outputs/RUN_COMPLETE.json').exists())

    def test_fast_conversion_matches_original_geometry_converter(self):
        from .geometry import contour_labels,stack_basis
        from .convert_cache import convert as original
        from .convert_population_cache import convert as fast
        path=self.case('conversion')
        with np.load(path) as z:r={k:np.array(z[k]) for k in z.files}
        rng=np.random.default_rng(81);basis=stack_basis(r['axis']);q=[];sdf=[]
        for i,pos in enumerate(r['s']):
            points=np.c_[rng.uniform(-.25,.25,(512,2)),np.full(512,pos)].astype(np.float32)
            region,de,di=contour_labels(points,r['boundary'][i],r['boundary_endo'][i],basis)
            distance=np.minimum(abs(de),abs(di));q.append(points);sdf.append(np.where(region==0,-distance,distance))
        source=self.root/'source.npz';np.savez_compressed(source,**{k:r[k] for k in ('boundary','boundary_endo','s','axis','center_mm','scale_mm','phase','case_id')},q=np.stack(q),sdf2d=np.stack(sdf))
        a=self.root/'old.npz';b=self.root/'fast.npz';original(source,a);fast(source,b)
        with np.load(a) as old,np.load(b) as new:
            self.assertEqual(set(old.files),set(new.files))
            for k in old.files:np.testing.assert_array_equal(old[k],new[k])

    def test_visit_sampler_coverage_survives_reloading(self):
        p=self.case('sampling');pool=self.root/'pool.npz'
        from .area_pool import build_pool as shell_pool
        shell_pool(p,[0,3,6],pool,capacities=(512,256,256))
        points=[]
        for visit in range(8):
            sampler=VisitSampler(pool);sampler.set_visit(visit,128,19);q,_=sampler.draw(0,128);points.append(q)
        self.assertEqual(len(np.unique(np.concatenate(points),axis=0)),1024)
        sampler=VisitSampler(pool);sampler.set_visit(3,128,19);q,_=sampler.draw(0,128)
        np.testing.assert_array_equal(q,points[3])

if __name__=='__main__':unittest.main()
