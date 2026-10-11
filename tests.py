import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from .model import AutoOccupancy,OccupancyDecoder,interpolate,loo
from .data import load_case,sample_plane
from .objective import objective
from .convert_cache import convert
from .make_synthetic import make_case
from .geometry import contour_report
from .extract_mesh import mesh_arrays


class Contracts(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19); torch.set_num_threads(2)

    def test_nonuniform_distance_and_gradient(self):
        s=torch.tensor([0.,.1,.3]); u=torch.tensor([[0.],[10.],[30.]])
        q=torch.tensor([[0.,0.,.05],[0.,0.,.2]],requires_grad=True)
        value=interpolate(q,s,u,torch.tensor([0.,0.,1.]))
        torch.testing.assert_close(value,torch.tensor([[5.],[20.]]))
        value.sum().backward(); torch.testing.assert_close(q.grad[:,2],torch.tensor([100.,100.]))
        torch.testing.assert_close(loo(s,u),torch.tensor([[10.]]))
        with self.assertRaises(ValueError): interpolate(torch.tensor([[0.,0.,-.1]]),s,u,torch.tensor([0.,0.,1.]))

    def test_decoder_depth_and_input_skip(self):
        d=OccupancyDecoder(); self.assertEqual(sum(isinstance(x,torch.nn.Linear) for x in d.modules()),5)
        self.assertEqual(sum(isinstance(x,torch.nn.Softplus) for x in d.modules()),4)
        self.assertEqual(d.b[0].in_features,256+132)

    def test_cache_and_supervision(self):
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'source.npz'; dst=Path(tmp)/'occ.npz'; make_case(src)
            convert(src,dst); c=load_case(dst,observed=[0,3,6])
            q,y,_=sample_plane(c,0,1024); self.assertEqual(int(y.sum()),512)
            self.assertEqual(len(c['s']),3)
            with self.assertRaises(FileExistsError): convert(src,dst)
            # Filled Epi labels must never be accepted as shell occupancy.
            with np.load(src) as z: r={k:z[k] for k in z.files}
            r['sdf2d']=-np.abs(r['sdf2d']); bad=Path(tmp)/'bad.npz'; np.savez(bad,**r)
            with self.assertRaises(ValueError): convert(bad,Path(tmp)/'bad_occ.npz')

    def test_direct_loo_seam_gradient_and_frozen_model(self):
        import json
        cfg=json.loads(Path(__file__).with_name('config.json').read_text())
        cfg.update(points=128,seam_weight=.001,seam_start=0,seam_points=8)
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'s.npz'; dst=Path(tmp)/'o.npz'; make_case(src); convert(src,dst)
            c=load_case(dst,observed=[0,3,6]); model=AutoOccupancy(width=32)
            code=torch.nn.Parameter(torch.randn(128)*.01)
            loss,stats=objective(model,code,c,cfg,0); loss.backward()
            self.assertGreater(stats['loo'],0)
            self.assertTrue(torch.isfinite(code.grad).all()); self.assertGreater(float(code.grad.norm()),0)
            for name in ('generator','decoder'):
                self.assertGreater(sum(float(x.grad.norm()) for x in getattr(model,name).parameters()),0)
            before={k:v.clone() for k,v in model.state_dict().items()}
            model.zero_grad(set_to_none=True); model.requires_grad_(False); code.grad=None
            opt=torch.optim.Adam([code],lr=.01)
            loss,_=objective(model,code,c,cfg,0); loss.backward(); opt.step()
            for k,v in model.state_dict().items(): torch.testing.assert_close(v,before[k],rtol=0,atol=0)
            self.assertTrue(all(x.grad is None for x in model.parameters()))

    def test_heldout_labels_do_not_affect_selected_objective(self):
        import json
        cfg=json.loads(Path(__file__).with_name('config.json').read_text()); cfg['points']=64
        with tempfile.TemporaryDirectory() as tmp:
            src=Path(tmp)/'s.npz'; dst=Path(tmp)/'o.npz'; make_case(src); convert(src,dst)
            with np.load(dst) as z: r={k:z[k] for k in z.files}
            selected=[0,3,6]; hidden=[1,2,4,5]
            r['occ'][hidden]=0; r['region'][hidden]=2
            alt=Path(tmp)/'altered.npz'; np.savez(alt,**r)
            c=load_case(dst,observed=selected); c2=load_case(alt,observed=selected)
            model=AutoOccupancy(width=32); code=torch.randn(128)*.01
            torch.manual_seed(99); loss,_=objective(model,code,c,cfg,0)
            torch.manual_seed(99); loss2,_=objective(model,code,c2,cfg,0)
            torch.testing.assert_close(loss,loss2,rtol=0,atol=0)

    def test_positive_occupancy_mesh_and_ring_closure(self):
        x=np.linspace(-1,1,40); grid=np.stack(np.meshgrid(x,x,x,indexing='ij'),-1)
        field=.6-np.linalg.norm(grid,axis=-1)
        v,f,report=mesh_arrays(field,np.full(3,-1.),np.full(3,1.),np.eye(3),np.array([10,20,30]),100.)
        self.assertTrue(report['watertight_edge_test']); self.assertEqual(report['boundary_edges'],0)
        self.assertLess(abs(np.linalg.norm(v-np.array([10,20,30]),axis=-1).mean()-60),1.)
        # Default marching cubes winding for positive-inside field must be outward.
        tri=v[f]-np.array([10,20,30]); normal=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        self.assertGreater(float(np.mean(np.sum(normal*tri.mean(1),axis=-1))),0)
        xx,yy=np.meshgrid(x,x,indexing='ij'); r=np.sqrt(xx**2+yy**2)
        report=contour_report(np.minimum(.7-r,r-.3)); self.assertEqual(report['closed_curves'],2)
        self.assertEqual(report['open_curves'],0)


if __name__ == '__main__': unittest.main()
