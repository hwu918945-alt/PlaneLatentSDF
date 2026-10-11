import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from torch import nn
from ..model import AutoOccupancy,interpolate,loo
from ..objective import objective
from ..area_pool import AreaSampler,build_pool,digest
from ..data import load_case
from ..inference import bounds,plane_grid
from ..chunked import backward_case
from ..repair_tests import fixture
from .model_case_only import CaseOnlyAutoOccupancy
from .nearest import nearest_plane_latent,query_variant
from .train_case_only import initialize,train
from .restore_variants import restore_variant


def config():return json.loads(Path(__file__).with_name('config_A_case_only.json').read_text())


class Ablations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2);cls.tmp=tempfile.TemporaryDirectory();cls.root=Path(cls.tmp.name)
        cls.case=cls.root/'case.npz';fixture(cls.case);cls.pool=cls.root/'pool.npz'
        build_pool(cls.case,[0,3,6],cls.pool,capacities=(8192,4096,4096))
    @classmethod
    def tearDownClass(cls):cls.tmp.cleanup()

    def test_decoder_is_same_class_and_architecture_parameter_counts(self):
        a=CaseOnlyAutoOccupancy();c=AutoOccupancy()
        self.assertIs(type(a.decoder),type(c.decoder));self.assertEqual(set(a.decoder.state_dict()),set(c.decoder.state_dict()))
        for k,v in a.decoder.state_dict().items():self.assertEqual(v.shape,c.decoder.state_dict()[k].shape)
        self.assertEqual(sum(isinstance(x,nn.Linear) for x in a.modules()),5)
        soft=[x for x in a.modules() if isinstance(x,nn.Softplus)];self.assertEqual(len(soft),4);self.assertTrue(all(x.beta==10 for x in soft))
        self.assertEqual(a.decoder.a[0].in_features,132);self.assertEqual(a.decoder.a[0].out_features,256)
        self.assertEqual(a.decoder(torch.randn(9,3),torch.randn(9,128),0).shape,(9,))
        self.assertEqual(sum(p.numel() for p in c.parameters())-sum(p.numel() for p in a.parameters()),49792)
        self.assertFalse(any('generator' in name for name,_ in a.named_parameters()))

    def test_constant_case_condition_accumulates_to_one_code(self):
        a=CaseOnlyAutoOccupancy();code=nn.Parameter(torch.randn(128));s=torch.tensor([0.,.3,.6]);u=a.plane_latents(code,s,0)
        for row in u:self.assertTrue(torch.equal(row,code))
        u.sum().backward();torch.testing.assert_close(code.grad,torch.full_like(code,3.))
        q=torch.tensor([[0.,0.,-.01]])
        with self.assertRaises(ValueError):a.field(q,s,u,torch.tensor([0.,0.,1.]),0)

    def test_nearest_and_interpolation_analytic_midpoint_ties(self):
        s=torch.tensor([0.,.3,.6]);u=torch.tensor([[0.],[3.],[6.]]);axis=torch.tensor([0.,0.,1.])
        heights=torch.tensor([0.,.1,.15,.2,.3,.4,.45,.5,.6]);q=torch.stack([torch.zeros_like(heights),torch.zeros_like(heights),heights],1)
        actual=nearest_plane_latent(q,s,u,axis).flatten()
        torch.testing.assert_close(actual,torch.tensor([0.,0.,0.,3.,3.,3.,3.,6.,6.]))
        torch.testing.assert_close(interpolate(q,s,u,axis).flatten(),heights*10,atol=5e-7,rtol=1e-6)
        for h in [-.001,.601]:
            for operator in [nearest_plane_latent,interpolate]:
                with self.assertRaises(ValueError):operator(torch.tensor([[0.,0.,h]]),s,u,axis)
        # Non-uniform distances are physical heights, not array-index weights.
        unequal=torch.tensor([0.,.2,.7]);q=torch.tensor([[0.,0.,.45]])
        self.assertEqual(float(nearest_plane_latent(q,unequal,u,axis)),3.)
        torch.testing.assert_close(interpolate(q,unequal,u,axis),torch.tensor([[4.5]]))

    def test_A_4096_chunks_match_full_gradient_and_equivalent_LOO_mix(self):
        cfg=config();cfg.update(width=32,points=8192,microbatch_points=4096)
        c=load_case(self.case,observed=[0,3,6]);c2=copy.copy(c);c3=copy.copy(c)
        for case in [c,c2,c3]:case['_area_sampler']=AreaSampler(self.pool)
        torch.manual_seed(19);m=CaseOnlyAutoOccupancy(width=32);m2=copy.deepcopy(m);m3=copy.deepcopy(m)
        code=nn.Parameter(torch.randn(128)*.01);z2=nn.Parameter(code.detach().clone());z3=nn.Parameter(code.detach().clone())
        value,stats=objective(m,code,c,cfg,0);value.backward()
        other,cs=backward_case(m2,z2,c2,cfg,0)
        mixed=dict(cfg,loo_mix=.25,loo_weight=1.)
        mix,ms=backward_case(m3,z3,c3,mixed,0)
        torch.testing.assert_close(value.detach(),other,rtol=2e-6,atol=3e-7);torch.testing.assert_close(other,mix,rtol=2e-6,atol=3e-7)
        for a,b,d in zip([code]+list(m.parameters()),[z2]+list(m2.parameters()),[z3]+list(m3.parameters())):
            torch.testing.assert_close(a.grad,b.grad,atol=2e-7,rtol=3e-4);torch.testing.assert_close(b.grad,d.grad,atol=2e-7,rtol=3e-4)
        self.assertEqual(cs['decoder_queries'],3*8192);self.assertEqual(cs['plane_loo'],[]);self.assertEqual(cs['loo'],0.)
        self.assertEqual(ms['decoder_queries'],4*8192)
        self.assertAlmostEqual(float(other),1.5*cs['direct']+cfg['latent_weight']*cs['latent'],places=6)

    def test_step0_pairing_and_reject_trained_initialization(self):
        cfg=config();c=load_case(self.case,observed=[0,3,6]);torch.manual_seed(19);m=AutoOccupancy();z=torch.randn(1,128)*.01
        payload=dict(format='plane_auto_occ_v1',step=0,model=m.state_dict(),codes=z,config=cfg,
            records=[dict(source_sha256=digest(self.case),observed=[0,3,6])],optimizer={'this must not be loaded':True})
        path=self.root/'initial_C.pt';torch.save(payload,path)
        a,code,metadata=initialize(cfg,c,step0=path)
        for k,v in a.decoder.state_dict().items():self.assertTrue(torch.equal(v,m.decoder.state_dict()[k]))
        self.assertTrue(torch.equal(code,z));self.assertTrue(metadata['decoder_code_exactly_paired']);self.assertFalse(metadata['optimizer_copied'])
        payload['step']=2000;torch.save(payload,path)
        with self.assertRaises(ValueError):initialize(cfg,c,step0=path)
        a,code,metadata=initialize(cfg,c);self.assertFalse(metadata['decoder_code_exactly_paired'])

    def test_hidden_contours_cannot_change_A_input_or_gradient(self):
        altered=self.root/'hidden_changed.npz';fixture(altered,hidden_shift=.5);pool=self.root/'hidden_pool.npz'
        build_pool(altered,[0,3,6],pool,capacities=(8192,4096,4096))
        cfg=config();cfg.update(points=128,microbatch_points=32,width=32)
        cases=[load_case(p,observed=[0,3,6]) for p in [self.case,altered]]
        np.testing.assert_array_equal(bounds(cases[0])[0],bounds(cases[1])[0]);np.testing.assert_array_equal(bounds(cases[0])[1],bounds(cases[1])[1])
        models=[];codes=[];losses=[]
        for c,p in zip(cases,[self.pool,pool]):
            m,z,_=initialize(cfg,c);c['_area_sampler']=AreaSampler(p)
            value,stats=backward_case(m,z[0],c,cfg,0);models.append(m);codes.append(z);losses.append(value)
        self.assertTrue(torch.equal(losses[0],losses[1]));self.assertTrue(torch.equal(codes[0].grad,codes[1].grad))
        for a,b in zip(models[0].parameters(),models[1].parameters()):self.assertTrue(torch.equal(a.grad,b.grad))

    def test_one_clip_and_adam_per_update_no_loo_and_explicit_A_format(self):
        cfg=config();cfg.update(points=128,microbatch_points=32,width=32,steps=3,pool_refresh_updates=2,pool_capacities=[8192,4096,4096])
        original_step=torch.optim.Adam.step;counter=[0]
        def counted(opt,*args,**kwargs):counter[0]+=1;return original_step(opt,*args,**kwargs)
        with patch.object(torch.optim.Adam,'step',counted),patch('torch.nn.utils.clip_grad_norm_',wraps=torch.nn.utils.clip_grad_norm_) as clip,patch('experiments.plane_auto_occ.chunked.loo',side_effect=AssertionError('A must never call LOO')):
            checkpoint=train(self.case,[0,3,6],self.pool,cfg,self.root/'tiny_A')
        self.assertEqual(counter[0],3);self.assertEqual(clip.call_count,3)
        ck=torch.load(checkpoint,weights_only=False);self.assertEqual(ck['format'],'plane_auto_occ_case_only_v1');self.assertEqual(ck['step'],3)
        model,code,u,obs,full=restore_variant(checkpoint,self.case,'case_only')
        self.assertTrue(torch.equal(u[0],u[-1]))
        with self.assertRaises(ValueError):restore_variant(checkpoint,self.case,'linear')
        rows=[json.loads(x) for x in (checkpoint.parent/'history.jsonl').read_text().splitlines()]
        self.assertEqual([r['pool_generation'] for r in rows],[0,0,1]);self.assertTrue(all(r['decoder_queries']==384 for r in rows))

    def test_BC_observed_grid_identical_and_state_read_only(self):
        torch.manual_seed(19);model=AutoOccupancy(width=32);code=torch.randn(128)*.01;c=load_case(self.case,observed=[0,3,6]);u=model.plane_latents(code,c['s'],c['phase'])
        original={k:v.clone() for k,v in model.state_dict().items()};ulo=u.detach().clone();lower,upper,basis=bounds(c,5)
        for pos in c['s'].numpy():
            q=plane_grid(float(pos),lower,upper,basis,16);linear=query_variant(model,q,c,u,'linear');nearest=query_variant(model,q,c,u,'nearest')
            self.assertLessEqual(float(abs(linear-nearest).max()),1e-5);np.testing.assert_array_equal(linear>=0,nearest>=0)
        for k,v in model.state_dict().items():self.assertTrue(torch.equal(v,original[k]))
        self.assertTrue(torch.equal(u,ulo))

    def test_native_uncapped_mesh_edge_report_with_numpy2_inverse_shape(self):
        from .evaluate_three import mesh_report
        n=32;lo=np.array([-.4,-.4,0.]);hi=np.array([.4,.4,.6]);x,y,z=np.meshgrid(*[np.linspace(lo[i],hi[i],n) for i in range(3)],indexing='ij')
        volume=(.35**2-x*x-y*y-(z-.3)**2).astype(np.float32)
        obs=dict(center_mm=np.zeros(3),scale_mm=100.,s=torch.tensor([0.,.3,.6]))
        report,arrays=mesh_report(volume,lo,hi,np.eye(3),obs,self.root/'sphere.obj')
        self.assertGreater(report['boundary_edges'],0);self.assertEqual(report['connected_components'],1)
        self.assertEqual(report['boundary_components'],2);self.assertTrue(report['boundary_vertices_all_degree2'])
        self.assertEqual(report['boundary_edge_locations']['axial'],report['boundary_edges'])
        self.assertEqual(report['boundary_edge_locations']['lateral'],0);self.assertEqual(report['boundary_edge_locations']['other'],0)
        self.assertTrue(report['no_artificial_caps'])

    def test_precision_adapter_preserves_model_math_weights_and_native_xyz_rounding(self):
        from .precision import promote
        torch.manual_seed(19);model=AutoOccupancy(width=32);c=load_case(self.case,observed=[0,3,6]);z=torch.randn(128)*.01;u=model.plane_latents(z,c['s'],0)
        state={k:v.detach().clone() for k,v in model.state_dict().items()}
        wrapper,code,latents,obs,full=promote((model,z,u,c,c))
        for k,v in wrapper.state_dict().items():self.assertTrue(torch.equal(v,state[k].double()))
        self.assertTrue(torch.equal(latents,u.double()));self.assertTrue(torch.equal(code,z.double()))
        q=torch.tensor([[.1,.2,.15],[.2,.1,.45]],dtype=torch.float32)
        expected=model.field(q.double(),obs['s'],latents,obs['axis'],0)
        torch.testing.assert_close(wrapper.field(q,obs['s'],latents,obs['axis'],0),expected,rtol=0,atol=0)

    def test_A_config_changes_only_required_fields(self):
        source=json.loads(Path(__file__).parents[1].joinpath('config.json').read_text());a=config()
        changes={k for k,v in a.items() if k not in source or source[k]!=v}
        self.assertEqual(changes,{'model_variant','loo_mix','loo_weight'});self.assertEqual(a['data_weight'],1.5);self.assertEqual(a['seam_weight'],0)
        self.assertEqual(a['steps'],2000);self.assertEqual(a['points'],32768);self.assertEqual(a['microbatch_points'],4096)

if __name__=='__main__':unittest.main()
