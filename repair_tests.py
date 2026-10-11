"""Repair contracts: exact geometry, coverage, no hidden input, gradient equivalence."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from .area_pool import PlaneClassifier,build_pool,AreaSampler
from .geometry import polygon_sdf,stack_basis
from .data import load_case
from .model import AutoOccupancy
from .objective import objective,mix_plane_losses
from .chunked import backward_case


def fixture(path,hidden_shift=0.):
    s=np.linspace(0,.6,7,dtype=np.float32); angle=np.linspace(0,2*np.pi,128,endpoint=False)
    epi=np.stack([np.c_[.2*np.cos(angle),.2*np.sin(angle),np.full(128,z)] for z in s]).astype(np.float32)
    endo=np.stack([np.c_[.19*np.cos(angle),.19*np.sin(angle),np.full(128,z)] for z in s]).astype(np.float32)
    epi[[1,2,4,5],:,0]+=hidden_shift;endo[[1,2,4,5],:,0]+=hidden_shift
    q=np.stack([np.c_[np.linspace(-.3,.3,8),np.zeros(8),np.full(8,z)] for z in s]).astype(np.float32)
    shape=q.shape[:2]
    np.savez_compressed(path,q=q,occ=np.zeros(shape,np.float32),region=np.full(shape,2,np.int64),near=np.zeros(shape,bool),
        boundary=epi,boundary_endo=endo,s=s,axis=np.array([0,0,1],np.float32),center_mm=np.zeros(3),scale_mm=np.array(100.),
        phase=np.array(0),case_id=np.array('fixture'),schema_version=np.array('plane_auto_occ_v1'))


class Repair(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.case=cls.root/'case.npz';fixture(cls.case);cls.pool=cls.root/'pool.npz'
        build_pool(cls.case,[0,3,6],cls.pool,capacities=(512,256,256),seed=19)

    @classmethod
    def tearDownClass(cls): cls.tmp.cleanup()

    def test_exact_band_classifier(self):
        angle=np.linspace(0,2*np.pi,128,endpoint=False)
        epi=np.c_[20*np.cos(angle),20*np.sin(angle)];endo=np.c_[19*np.cos(angle),19*np.sin(angle)]
        uv=np.random.default_rng(51).uniform(-23,23,(2048,2))
        classifier=PlaneClassifier(epi,endo);r,valid,near=classifier.classify(uv)
        de=polygon_sdf(uv,epi);di=polygon_sdf(uv,endo);dist=np.minimum(np.abs(de),np.abs(di))
        np.testing.assert_array_equal(r,np.where(di<0,1,np.where(de<0,0,2)))
        np.testing.assert_array_equal(valid,dist>.05);np.testing.assert_array_equal(near,dist<=3.)

    def test_no_replacement_and_complete_eight_step_coverage(self):
        sampler=AreaSampler(self.pool);seen=[]
        for _ in range(8):
            q,y=sampler.draw(0,128);self.assertEqual(int(y.sum()),64);self.assertEqual(len(np.unique(q,axis=0)),128);seen.append(q)
        self.assertEqual(len(np.unique(np.concatenate(seen),axis=0)),1024)
        self.assertTrue(all(v['coverage_fraction']==1 for v in sampler.last_stats[0]))
        sampler.draw(0,128);self.assertTrue(all(v['completed_cycles']==1 for v in sampler.last_stats[0]))
        # A thin ring has no far positives; its class quota still comes from unique near points.
        self.assertEqual(int((~sampler.near[0,:512]).sum()),0)
        with self.assertRaises(ValueError):sampler.draw(0,2048)

    def test_hidden_contours_cannot_change_pool(self):
        alternate=self.root/'changed_hidden.npz';fixture(alternate,hidden_shift=.5)
        path=self.root/'changed_hidden_pool.npz';build_pool(alternate,[0,3,6],path,capacities=(512,256,256),seed=19)
        with np.load(self.pool) as a,np.load(path) as b:
            for k in ('q','region','near','s','axis','boundary','boundary_endo'):
                np.testing.assert_array_equal(a[k],b[k])

    def test_plane_weight_conservation(self):
        for beta in (0.,.25,1.):
            d=[torch.tensor(float(i+1),requires_grad=True) for i in range(3)]
            value=mix_plane_losses(d,{1:d[1]},beta,1.5);value.backward()
            for v in d:self.assertEqual(float(v.grad),.5)

    def test_full_and_chunked_gradient_equivalence(self):
        cfg=json.loads(Path(__file__).with_name('config.json').read_text());cfg.update(points=128,microbatch_points=32)
        c=load_case(self.case,observed=[0,3,6]);c2=copy.copy(c)
        c['_area_sampler']=AreaSampler(self.pool);c2['_area_sampler']=AreaSampler(self.pool)
        torch.manual_seed(19);m=AutoOccupancy(width=32);m2=copy.deepcopy(m)
        z=torch.nn.Parameter(torch.randn(128)*.01);z2=torch.nn.Parameter(z.detach().clone())
        loss,stats=objective(m,z,c,cfg,0);loss.backward()
        value,chunkstats=backward_case(m2,z2,c2,cfg,0)
        torch.testing.assert_close(loss.detach(),value,atol=2e-7,rtol=2e-6)
        for a,b in zip([z]+list(m.parameters()),[z2]+list(m2.parameters())):
            torch.testing.assert_close(a.grad,b.grad,atol=2e-7,rtol=3e-4)
        self.assertEqual(chunkstats['independent_points'],384);self.assertEqual(chunkstats['decoder_queries'],512)
        # The regularizer and all parameter gradients enter the same single global clip/Adam update.
        for network,code in ((m,z),(m2,z2)):
            opt=torch.optim.Adam([{'params':[code],'lr':.001},{'params':network.parameters(),'lr':.001}])
            torch.nn.utils.clip_grad_norm_([code]+list(network.parameters()),5,error_if_nonfinite=True);opt.step()
        for a,b in zip([z]+list(m.parameters()),[z2]+list(m2.parameters())):
            torch.testing.assert_close(a,b,atol=2e-6,rtol=2e-5)

    def test_refresh_after_ten_full_coverages(self):
        import subprocess,sys
        cfg=json.loads(Path(__file__).with_name('config.json').read_text())
        cfg.update(points=128,microbatch_points=32,width=32,pool_capacities=[512,256,256])
        path=self.root/'small_config.json';path.write_text(json.dumps(cfg));out=self.root/'refresh_fit'
        result=subprocess.run([sys.executable,'-B','-m','experiments.plane_auto_occ.train',
            '--case',str(self.case),'--observed','0,3,6','--area-pool',str(self.pool),
            '--config',str(path),'--steps','81','--device','cpu','--output',str(out)],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stderr)
        rows=[json.loads(v) for v in (out/'history.jsonl').read_text().splitlines()]
        self.assertEqual([r['step'] for r in rows],list(range(1,82)))
        self.assertEqual([r['pool_generation'] for r in rows],[0]*80+[1])
        self.assertTrue(all(v['completed_cycles']==10 for plane in rows[79]['sampling'] for v in plane))
        self.assertTrue(all(v['coverage_fraction']==.125 for plane in rows[80]['sampling'] for v in plane))
        updated=out/'sampling_pools/generation_001.npz'
        with np.load(self.pool) as a,np.load(updated) as b:
            self.assertFalse(np.array_equal(a['q'],b['q']))
            self.assertEqual(json.loads(str(b['metadata_json']))['observed'],[0,3,6])


if __name__=='__main__':unittest.main()
