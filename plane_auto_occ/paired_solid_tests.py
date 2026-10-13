"""Topology, real-segment pair units, and solid bounds regression checks."""
import unittest
import numpy as np
import torch
from .paired_solid_geometry import SolidPlane
from .paired_pool_coordinates import repair_coordinates,map_world
from .geometry import stack_basis
from .inference import bounds


def ring(center,radius,vertices=64):
    t=np.arange(vertices)*2*np.pi/vertices
    return np.column_stack([np.cos(t),np.sin(t)])*radius+center


class SolidPairs(unittest.TestCase):
    def test_subband_spur_rejection_is_audited_and_preserves_scale_quotas(self):
        spur=np.array([[0,0],[5,0],[5,5],[2.52,5],[2.5,5.15],[2.48,5],[0,5]],float)
        plane=SolidPlane([spur],[-3,-3],[8,8])
        bank=plane.pairs(0,512,np.random.default_rng(19))
        self.assertGreater(len(bank['unsupported_anchors_rejected']),0)
        np.testing.assert_array_equal(np.bincount(bank['scale_id'],minlength=3),[257,153,102])
        labels,valid,d=plane.classify(bank['q'].reshape(-1,2))
        np.testing.assert_array_equal(labels,np.tile([0,1],512));self.assertTrue(valid.all())
        self.assertGreater(d.min(),.05)

    def test_holes_components_real_segments_and_pair_sides(self):
        rings=[ring([-5,0],2),ring([-5,0],.5),ring([5,0],2)]
        plane=SolidPlane(rings,[-10,-5],[10,5])
        labels,valid,_=plane.classify(np.array([[-5,0],[-4,0],[5,0],[0,0]],float))
        np.testing.assert_array_equal(labels,[1,0,0,1]);self.assertTrue(valid.all())
        c,v,_,_,cum=plane.edges[0]
        self.assertEqual(len(v),192)
        for i,r in enumerate(rings):
            np.testing.assert_allclose(c[i*64:(i+1)*64]+v[i*64:(i+1)*64],np.roll(r,-1,axis=0),atol=1e-12)
        rng=np.random.default_rng(19)
        for b in [0,1]:
            bank=plane.pairs(b,512,rng)
            r,valid,d=plane.classify(bank['q'].reshape(-1,2))
            np.testing.assert_array_equal(r,np.tile([0,1],512))
            self.assertTrue(valid.all());self.assertGreater(d.min(),.05)
            self.assertTrue((bank['delta']<=.4*bank['clearance']+1e-7).all())

    def test_float32_repair_preserves_binary_solid_banks(self):
        plane=SolidPlane([ring([0,0],3)],[-6,-6],[6,6]);rng=np.random.default_rng(19)
        banks=[plane.pairs(b,64,rng) for b in [0,1]]
        uv=np.concatenate([bank['q'].reshape(-1,2) for bank in banks]+[plane.uniform(160,rng)])
        source=np.r_[np.tile([0,1],64),np.tile([2,3],64),np.full(160,4)].astype(np.int8)
        basis=stack_basis([0,0,1]);q=map_world(uv,10.,basis,.1)
        # A deliberately corrupted final coordinate triggers a same-anchor pair retry.
        q[128]=q[0]
        repaired,r,audit=repair_coordinates(plane,q,source,banks,10.,basis,.1,rng)
        self.assertTrue(audit['boundary_pairs_retried'])
        np.testing.assert_array_equal(r[source<4],np.array([0,1,0,1])[source[source<4]])
        self.assertEqual(len(np.unique(repaired,axis=0)),len(q))
        self.assertTrue(all(a['anchor_preserved'] for a in audit['boundary_pairs_retried']))

    def test_variable_ring_geometry_reference_bounds(self):
        rings=[[ring([0,0],2)],[ring([20,0],3),ring([-10,0],1)],[ring([2,0],4)]]
        heights=torch.tensor([-.2,0,.4]);scale=10.;center=np.array([1.,2.,0.])
        points=np.concatenate([np.c_[(np.concatenate(r)-center[:2])/scale,np.full(sum(len(x) for x in r),float(z))]
                               for r,z in zip(rings,heights)])
        c=dict(schema='plane_auto_occ_solid_v1',axis=torch.tensor([0.,0.,1.]),boundary=torch.zeros(3,1,3),
               geometry_points=torch.tensor(points),solid_rings_mm=rings,s=heights,observed=[0,1,2],
               center_mm=center,scale_mm=scale)
        full=bounds(c,20.)
        referenced=bounds(dict(c,_roi_reference_observed=[0,1,2]),20.)
        for a,b in zip(full,referenced):np.testing.assert_array_equal(a,b)
        limited=bounds(dict(c,_roi_reference_observed=[0,2]),20.)
        self.assertLess((limited[1]-limited[0])[:2].prod(),(full[1]-full[0])[:2].prod())


if __name__=='__main__':unittest.main()
