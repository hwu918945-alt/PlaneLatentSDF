"""Behavioral checks for shell geometry, information paths and optimizer safety."""
import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from prepare_cache import convert
from core.data import load_case, require_shell, polygon_sdf
from core.geometry import shell_sdf, validate_shell
from core.sampling import (POOL_COUNTS, ENCODER_COUNTS, SUPERVISION_COUNTS,
                           stratified_indices, encoder_view, supervision_view,
                           encode_case, eikonal_queries, deterministic_generator)
from core.model import PlaneField, make_teacher, ema_update, interpolate_latent, leave_one_out_latent
from core.training import case_objective
from core.losses import coordinate_gradient, eikonal_loss, RealPlaneDistillation, seam_normal_loss
from core.mesh import volume_to_mesh, stack_basis
from core.evaluation import case_geometry, aggregate_geometry


def fixture_record(cid='synthetic', phase='ED', split='train', heights=(0., 7., 30., 42.)):
    angle = np.arange(64)/64*2*np.pi
    planes = []
    for i, height in enumerate(heights):
        outer = 25. + 2*np.sin(i); inner = outer-6.
        if phase == 'ES':
            outer *= .9; inner *= .8
        epi = np.stack((outer*np.cos(angle), .8*outer*np.sin(angle)), -1)
        endo = np.stack((inner*np.cos(angle), .8*inner*np.sin(angle)), -1)
        planes.append(dict(origin_mm=[12., -9., height+40.], u=[1.,0.,0.], v=[0.,1.,0.],
                           epi_uv_mm=epi.tolist(), endo_uv_mm=endo.tolist(), source_slice_index=i))
    return dict(case_id=cid, phase=phase, split=split, anatomy='LV_myocardium_shell', planes=planes)


class ShellTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp = tempfile.TemporaryDirectory()
        cls.path = Path(cls.temp.name)/'case.npz'
        convert(fixture_record(), cls.path, np.random.default_rng(1))
        cls.case = load_case(cls.path)
        cls.cfg = json.loads((Path(__file__).resolve().parents[1]/'config.json').read_text())

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def test_exact_shell_signs_and_invalid_nesting(self):
        epi = np.array([[-2.,-2.], [2.,-2.], [2.,2.], [-2.,2.]])
        endo = epi/2
        d = shell_sdf(np.array([[0.,0.], [1.5,0.], [3.,0.], [1.,0.], [2.,0.]]), epi, endo)
        np.testing.assert_allclose(d, [1., -.5, 1., 0., 0.])
        with self.assertRaises(ValueError):
            validate_shell(endo, epi)
        with self.assertRaises(ValueError):
            validate_shell(epi, epi)

    def test_pools_quotas_labels_and_continuous_boundary(self):
        c = self.case
        for plane in c['sampling_source']:
            np.testing.assert_array_equal(np.bincount(plane.numpy()), POOL_COUNTS)
        self.assertEqual(c['q'].shape[1], 8192)
        for quotas in (ENCODER_COUNTS, SUPERVISION_COUNTS):
            indices = stratified_indices(c['sampling_source'], quotas)
            for i, ids in enumerate(indices):
                self.assertEqual(len(ids.unique()), sum(quotas))
                np.testing.assert_array_equal(np.bincount(c['sampling_source'][i, ids].numpy()), quotas)
        for q, d, boundary in supervision_view(c):
            self.assertEqual(len(q), 512); self.assertEqual(boundary.shape, (2,64,3))
        with np.load(self.path) as f:
            for tag in (0,2):
                self.assertTrue(np.all(f['sdf2d'][f['sampling_source']==tag] == 0))
        # Gaussian source cannot determine the true region or the SDF sign.
        values = c['sdf2d'][c['sampling_source']==1]
        self.assertTrue(bool((values<0).any() and (values>0).any()))
        require_shell(c)
        with self.assertRaises(ValueError):
            require_shell({'anatomy':'LV_epi_envelope'})
        changed = copy.deepcopy(fixture_record()); changed['anatomy'] = 'LV_epi_envelope'
        with self.assertRaises(ValueError):
            convert(changed, Path(self.temp.name)/'bad.npz', np.random.default_rng(1))

    def test_determinism_and_hidden_plane_not_in_loo(self):
        c = self.case; torch.manual_seed(6)
        model = PlaneField(**self.cfg['model'])
        rng = torch.get_rng_state().clone()
        first = encoder_view(c, True); second = encoder_view(c, True)
        self.assertTrue(torch.equal(first[0], second[0]))
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(first[0].shape[1], 2048)
        changed = dict(c); changed['q'] = c['q'].clone(); changed['sdf2d'] = c['sdf2d'].clone()
        changed['q'][1, :, :2] += 10; changed['sdf2d'][1] += 5
        before = leave_one_out_latent(c['s'], encode_case(model,c))[0]
        after = leave_one_out_latent(c['s'], encode_case(model,changed))[0]
        torch.testing.assert_close(before,after,rtol=0,atol=0)
        np.testing.assert_allclose(c['center_mm'], fixture_record()['planes'][0]['origin_mm'])

    def test_gap_sampling_distribution_rotated_and_complete_gradient(self):
        c = self.case
        q = eikonal_queries(c, 20000, deterministic_generator(c,12))
        h = q@c['axis']; intervals = (torch.searchsorted(c['s'],h,right=True)-1).clamp(0,len(c['s'])-2)
        expected = (c['s'][1:]-c['s'][:-1])/(c['s'][-1]-c['s'][0])
        observed = torch.bincount(intervals,minlength=len(expected))/len(h)
        torch.testing.assert_close(expected, observed, atol=.015, rtol=0)
        self.assertTrue(bool(((h>=c['s'][0])&(h<=c['s'][-1])).all()))
        rotation = torch.tensor(stack_basis(np.array([.2,.3,.9])),dtype=torch.float32)
        rotated = dict(c)
        for key in ('q','boundary','boundary_endo'):
            rotated[key] = c[key]@rotation.T
        rotated['axis'] = rotation[:,2]
        rq = eikonal_queries(rotated, 128, deterministic_generator(rotated,13))
        self.assertTrue(bool(((rq@rotated['axis']>=c['s'][0]-1e-6)&(rq@rotated['axis']<=c['s'][-1]+1e-6)).all()))
        axis = torch.tensor([0.,0.,1.],dtype=torch.float64)
        s = torch.tensor([0.,1.,4.],dtype=torch.float64)
        z = torch.tensor([[0.],[2.],[8.]],dtype=torch.float64)
        queries = torch.tensor([[.2,.1,.4]],dtype=torch.float64,requires_grad=True)
        field = lambda x:x[:,0]+interpolate_latent(x,s,z,axis)[:,0]
        gradient = coordinate_gradient(field(queries),queries)
        torch.testing.assert_close(gradient,torch.tensor([[1.,0.,2.]],dtype=torch.float64))
        numeric=[]
        for i in range(3):
            delta=torch.zeros_like(queries);delta[:,i]=1e-6
            numeric.append(float(((field(queries+delta)-field(queries-delta))/2e-6).detach()))
        np.testing.assert_allclose(numeric,gradient.detach()[0],atol=1e-8)
        torch.testing.assert_close(leave_one_out_latent(s,z), z[1:2])
        with self.assertRaises(ValueError):
            interpolate_latent(torch.tensor([[0.,0.,5.]],dtype=torch.float64),s,z,axis)

    def test_step_zero_eikonal_and_full_loss_backward_ema(self):
        c = self.case; torch.manual_seed(9)
        model = PlaneField(**self.cfg['model']); teacher=make_teacher(model)
        distill=RealPlaneDistillation(prototypes=self.cfg['model']['prototypes'])
        optimizer=torch.optim.AdamW(model.parameters(),lr=1e-3)
        for epoch in (0, 6):
            optimizer.zero_grad(set_to_none=True)
            loss,logs,logits=case_objective(model,teacher,distill,c,self.cfg,epoch)
            self.assertGreater(logs['weighted_eikonal'],0)
            if epoch==0:
                self.assertEqual(logs['ramp'],0.)
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(model.parameters(),5.,error_if_nonfinite=True)
            self.assertTrue(torch.isfinite(norm))
            for module in (model.encoder, model.decoder):
                grads=[p.grad for p in module.parameters() if p.grad is not None]
                self.assertGreater(sum(float(g.abs().sum()) for g in grads),0)
            if epoch==6:
                self.assertGreater(sum(float(p.grad.abs().sum()) for p in model.head.parameters()),0)
            self.assertTrue(all(p.grad is None for p in teacher.parameters()))
            before={n:p.clone() for n,p in teacher.named_parameters()}
            optimizer.step();distill.update_center(logits);ema_update(teacher,model,.9)
            for name,p in teacher.named_parameters():
                torch.testing.assert_close(p,.9*before[name]+.1*dict(model.named_parameters())[name],atol=1e-7,rtol=1e-5)
        model.zero_grad(set_to_none=True)
        z=model.encoder(*encoder_view(c),c['phase'])
        eik=eikonal_loss(lambda q:model.field(q,c['s'],z,c['axis'],c['phase']),eikonal_queries(c,128))
        eik.backward()
        self.assertGreater(float(model.encoder.point[0].weight.grad.abs().sum()),0)

    def test_extraction_uses_2048_and_retains_shell_without_caps(self):
        from extract_mesh import extract
        import trimesh
        c=self.case
        class AnalyticField:
            def encoder(self,q,d,phase):
                if q.shape[1]!=2048:raise AssertionError('Inference point budget mismatch')
                return c['s'][:,None]
            def field(self,q,s,z,axis,phase):
                r=(q-torch.tensor([0.,0.,.21])).norm(dim=-1)
                return torch.maximum(r-.27,.10-r)
        out=Path(self.temp.name)/'shell.obj'
        report=extract(AnalyticField(),c,out,resolution=40,chunk=512)
        self.assertEqual(report['anatomy'],'LV_myocardium_shell')
        self.assertFalse(report['artificial_axial_caps'])
        mesh=trimesh.load(out,process=False)
        self.assertEqual(len(mesh.split(only_watertight=False)),2)
        self.assertGreater(report['vertices_on_axial_boundary'],0)
        with self.assertRaises(ValueError):
            extract(AnalyticField(),c,out,resolution=40,cap_stack=True)

    def test_seam_and_mesh_preserve_both_components_and_cavity(self):
        axis=torch.tensor([0.,0.,1.])
        points=torch.tensor([[.01,0.,.2],[-.03,0.,.3]])
        self.assertLess(float(seam_normal_loss(lambda q:q[:,0]+.1*q[:,2],points,axis,.001)),1e-6)
        basis=stack_basis(np.array([.2,.3,.9]))
        size=64;grid=np.stack(np.meshgrid(*[np.linspace(-.8,.8,size)]*3,indexing='ij'),-1)
        radius=np.linalg.norm(grid,axis=-1)
        volume=np.maximum(radius-.5,.3-radius)
        vertices,faces,report=volume_to_mesh(volume.astype(np.float32),[-.8]*3,[.8]*3,basis,np.array([12.,-9.,40.]),100.)
        import trimesh
        mesh=trimesh.Trimesh(vertices,faces,process=False)
        self.assertEqual(len(mesh.split(only_watertight=False)),2)
        self.assertTrue(report['watertight_edge_test'])
        radial=np.linalg.norm(vertices-np.array([12.,-9.,40.]),axis=1)
        self.assertLess(float(np.minimum(abs(radial-50),abs(radial-30)).max()),.15)
        # Analytic annular field validates both surface metrics and shell Dice.
        c=self.case
        class Analytic:
            def encoder(self,q,d,phase):return c['s'][:,None]
            def decoder(self,q,z,phase):
                i=torch.argmin((z[:,:1]-c['s'][None]).abs(),dim=1)
                outer=25.+2*torch.sin(i.float());inner=outer-6.
                radial=torch.sqrt(q[:,0]**2+(q[:,1]/.8)**2)*100
                return torch.maximum(radial-outer,inner-radial)/100
        geometry=case_geometry(Analytic(),c,spacing_mm=.5)
        self.assertTrue(all(row['valid'] for row in geometry['direct']))
        summary=aggregate_geometry([geometry])
        self.assertGreater(summary['direct_contour_dice'],.98)
        self.assertLess(summary['direct_epi_mcd_mm'],.2)
        self.assertLess(summary['direct_endo_mcd_mm'],.2)


if __name__=='__main__':
    unittest.main()
