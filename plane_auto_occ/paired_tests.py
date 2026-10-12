"""Integration contracts, exact microbatch gradients, pool units, and restoration."""
import copy,json,tempfile,unittest
from pathlib import Path
import numpy as np
import torch
from .model import model_from_config,loo
from .data import load_case
from .area_pool import digest
from .paired_loss import shape_loss,backward_case,monitor
from .paired_pools import build,location,check,PairedSampler,PairedPoolStore
from .paired_suite import arm_config


def config(root,points=256):
    return dict(latent_dim=8,width=16,points=points,steps=2,seed=19,threads=2,lr_shared=.001,lr_latent=.001,
        clip_grad=5.,latent_weight=1e-4,loo_mix=.25,loo_weight=1.,direct_weight=.5,data_weight=1.5,
        seam_weight=0.,pool_refresh_updates=80,pool_margin_mm=20.,boundary_band_mm=.05,
        near_mm=3.,log_every=50,save_every=1,fast_query_points=71,checkpoint_seconds=300.)


def cache(path,patient='008',phase=0):
    t=np.linspace(0,2*np.pi,64,endpoint=False);s=np.array([-.2,0,.3],np.float32)
    epi=[];endo=[];q=[]
    for z in s:
        epi.append(np.column_stack([.5*np.cos(t),.5*np.sin(t),np.full(64,z)]))
        endo.append(np.column_stack([.3*np.cos(t),.3*np.sin(t),np.full(64,z)]))
        q.append(np.array([[.4,0,z],[0,0,z],[.7,0,z]],np.float32))
    region=np.tile(np.array([0,1,2],np.int64),(3,1))
    np.savez(path,q=np.array(q,np.float32),region=region,occ=(region==0).astype(np.float32),near=np.ones((3,3),bool),
        boundary=np.array(epi,np.float32),boundary_endo=np.array(endo,np.float32),s=s,axis=np.array([0,0,1],np.float32),
        center_mm=np.zeros(3),scale_mm=np.array(10.),phase=np.array(phase),case_id=np.array(patient),schema_version=np.array('plane_auto_occ_v1'))


class FixedSampler:
    def __init__(self,q,y,source):self.q=q;self.y=y;self.last_source={i:source[i] for i in range(len(q))};self.last_stats={i:{} for i in range(len(q))}
    def draw(self,i,count):return self.q[i],self.y[i]


class Integration(unittest.TestCase):
    def setUp(self):torch.set_num_threads(2)
    def test_whole_plane_dice_exact_chunk_gradients(self):
        rng=np.random.default_rng(19);q=rng.normal(size=(3,256,3));y=(rng.random((3,256))<.25).astype(np.float64)
        source=np.tile(np.r_[np.zeros(96,np.int8),np.full(160,4,np.int8)],(3,1))
        for arm in ('A','B','C','D'):
            torch.manual_seed(19);cfg=arm_config(config('/tmp'),arm,'/tmp',[0,1,2]);cfg['fast_query_points']=71
            m=model_from_config(cfg).double();ref=copy.deepcopy(m);z=torch.randn(8,dtype=torch.float64,requires_grad=True);zr=z.detach().clone().requires_grad_()
            c=dict(s=torch.tensor([-.2,0.,.3],dtype=torch.float64),phase=0,observed=[0,1,2],axis=torch.tensor([0,0,1]),_area_sampler=FixedSampler(q,y,source))
            value,_=backward_case(m,z,c,cfg,0)
            u=ref.plane_latents(zr,c['s'],0);ul=loo(c['s'],u);terms=[]
            for i in range(3):
                qt=torch.tensor(q[i]);gt=torch.tensor(y[i]);st=torch.tensor(source[i])
                direct=shape_loss(ref.decoder(qt,u[i].expand(256,-1),0),gt,st,cfg['dice_weight'])[0]
                if i==1:
                    leave=shape_loss(ref.decoder(qt,ul[0].expand(256,-1),0),gt,st,cfg['dice_weight'])[0]
                    terms.append(.75*direct+.25*leave)
                else:terms.append(direct)
            loss=1.5*torch.stack(terms).mean()+1e-4*zr.square().mean();loss.backward()
            self.assertAlmostEqual(float(value),float(loss.detach()),places=11)
            self.assertTrue(torch.allclose(z.grad,zr.grad,rtol=1e-9,atol=1e-11))
            for (name,a),(name2,b) in zip(m.named_parameters(),ref.named_parameters()):
                self.assertEqual(name,name2);self.assertTrue(torch.allclose(a.grad,b.grad,rtol=1e-9,atol=1e-11),name)

    def test_dice_uses_only_source_not_location(self):
        a=torch.tensor([9.,-9.,0.,0.],requires_grad=True);g=torch.tensor([0.,1.,0.,1.]);s=torch.tensor([0,1,4,4])
        _,p=shape_loss(a,g,s);gradient=torch.autograd.grad(p['dice_roi'],a)[0]
        self.assertTrue(torch.equal(gradient[:2],torch.zeros(2)))
        self.assertGreater(float(gradient[2:].abs().sum()),0)
    def test_monitor_has_no_intervention(self):
        g=torch.tensor([0.,1.]);s=torch.tensor([4,4]);a=torch.full((2,),-20.)
        r=monitor(a,g,s,0,'direct');self.assertTrue(r['constant_candidate']);self.assertTrue(r['empty_prediction'])
    def test_D_FULL_identical_and_decoder_parameter_keys(self):
        b=config('/tmp');d=arm_config(b,'D','/tmp',[0,1,2]);f=arm_config(b,'FULL','/tmp',[0,1,2]);self.assertEqual(d,f)
        torch.manual_seed(19);a=model_from_config(arm_config(b,'A','/tmp',[0,1,2]))
        torch.manual_seed(19);m=model_from_config(d)
        self.assertEqual(list(a.state_dict()),list(m.state_dict()))
        self.assertTrue(all(torch.equal(v,m.state_dict()[k]) for k,v in a.state_dict().items()))
        q=torch.randn(13,3);z=torch.randn(13,8);x=torch.cat([q,z,q[:,:1]*0],-1)
        self.assertTrue(torch.equal(a.decoder(q,z,0),a.decoder.b(torch.cat([a.decoder.a(x),x],-1)).squeeze(-1)))
    def test_pool_eight_visit_units_sources_and_contract(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'case.npz';cache(p);c=load_case(p);cfg=arm_config(config(td),'B',td,[0,1,2])
            build(p,None,Path(td)/'point_pools',cfg,0);path=location(Path(td)/'point_pools',c,cfg,0)
            m=check(path,c,cfg,0,full=True);sampler=PairedSampler(path,m);draws=[]
            for visit in range(8):
                sampler.set_visit(visit);q,y=sampler.draw(0,256);draws.append(q)
                source=sampler.last_source[0];self.assertEqual(int((source==4).sum()),160)
                self.assertTrue(np.array_equal(source[:48],np.tile([0,1],24)))
                self.assertTrue(np.array_equal(y[:48],np.tile([1,0],24)))
                self.assertTrue(np.array_equal(source[48:96],np.tile([2,3],24)))
            self.assertEqual(len(np.unique(np.concatenate(draws),axis=0)),2048)
            sampler.set_visit(0);self.assertTrue(np.array_equal(draws[0],sampler.draw(0,256)[0]))
            changed=dict(cfg,pair_scales_mm=[.5,.75,2.])
            with self.assertRaises(ValueError):check(path,c,changed,0)
            store=PairedPoolStore(Path(td)/'point_pools',cfg);static=store.get(c,0);self.assertTrue(np.array_equal(static.last_source[0],sampler.last_source[0]))
    def test_checkpoint_restore_matches_residual_forward(self):
        from .inference import restore
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'case.npz';cache(p);c=load_case(p);cfg=arm_config(config(td),'D',td,[0,1,2]);m=model_from_config(cfg)
            z=torch.randn(8)*.01;ck=Path(td)/'last.pt'
            torch.save(dict(format='plane_auto_occ_v1',model=m.state_dict(),codes=z[None],config=cfg,
                records=[dict(case_id=c['case_id'],phase=0,observed=[0,1,2],source_sha256=digest(p))]),ck)
            restored,code,u,obs,full=restore(ck,p)
            q=torch.tensor([[.4,0,0]],dtype=torch.float32)
            self.assertTrue(torch.equal(m.field(q,c['s'],m.plane_latents(z,c['s'],0),c['axis'],0),restored.field(q,obs['s'],u,obs['axis'],0)))


if __name__=='__main__':unittest.main()
