"""OReX Balloon Dog -> parallel multi-ring observations and frozen heldout set."""
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import trimesh
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'runtime'))
from core.data import load_case
from geometry_adapter import section_rings,all_boundary,region_sdf


def main():
    protocol=json.loads((ROOT/'protocol.json').read_text())
    mesh=trimesh.load(ROOT/'source/balloondog_scaled.stl',process=True)
    if not mesh.is_watertight or not mesh.is_winding_consistent:
        raise ValueError('Reference not suitable for a filled-region benchmark; do not silently repair')
    if len(mesh.split(only_watertight=False))!=1:raise ValueError('Unexpected reference components')
    original_bounds=mesh.bounds.copy()
    # Dimensionless source, explicit fixed benchmark conversion, no bbox fitting.
    mesh.vertices*=100.
    mesh.export(ROOT/'source/balloon_dog_reference_mm.ply')
    mesh.export(ROOT/'source/balloon_dog_reference_mm.obj')
    mesh=trimesh.load(ROOT/'source/balloon_dog_reference_mm.ply',process=False)
    low,high=mesh.bounds[:,2]+np.array([.75,-.75])
    all_heights=[];all_bounds=[];groups=[]
    for count in protocol['slice_counts']:
        group=ROOT/f'slices_{count:03d}'
        heights=np.linspace(low,high,count)
        rings=[section_rings(mesh,h) for h in heights]
        boundary=[all_boundary(r,256) for r in rings]
        world=np.asarray([np.c_[b,np.full(len(b),h)] for b,h in zip(boundary,heights)])
        center=world.reshape(-1,3).mean(0)
        q=[];d=[];rng=np.random.default_rng(protocol['training_seed'])
        for b,h,r in zip(boundary,heights,rings):
            seeds=b[rng.integers(len(b),size=768)]
            sigma=rng.choice([1.,3.,6.],size=768,p=[.4,.4,.2])
            near=seeds+rng.normal(size=seeds.shape)*sigma[:,None]
            xy=np.r_[near,rng.uniform(b.min(0)-10,b.max(0)+10,(256,2))]
            q.append((np.c_[xy,np.full(len(xy),h)]-center)/100.)
            d.append(region_sdf(xy,r)/100.)
        np.savez_compressed(group/'data/balloon_dog_observed.npz',q=np.asarray(q,np.float32),sdf2d=np.asarray(d,np.float32),
            boundary=np.asarray((world-center)/100,np.float32),s=np.asarray((heights-center[2])/100,np.float32),
            axis=np.array([0,0,1],np.float32),center_mm=center,scale_mm=np.array(100.),phase=np.array(0),case_id=np.array('balloon_dog'))
        planes=[dict(height_mm=float(h),rings_uv_mm=[r.tolist() for r in rr]) for h,rr in zip(heights,rings)]
        (group/'data/observed_contours.json').write_text(json.dumps(dict(planes=planes),separators=(',',':')))
        case=load_case(group/'data/balloon_dog_observed.npz')
        assert len(case['s'])==count
        assert float((case['s'][1:]-case['s'][:-1]).min())>4*.003
        all_bounds.append(world.reshape(-1,3));all_heights.extend(heights.tolist())
        report=dict(case_id='balloon_dog',observed_planes=count,spacing_mm=float(np.diff(heights).mean()),
            extent_mm=[float(heights[0]),float(heights[-1])],scale_mm=100,center_mm=center.tolist(),
            boundary_ring_counts=[len(p['rings_uv_mm']) for p in planes],reference_mesh_used_for_loss=False)
        (group/'data/preparation_report.json').write_text(json.dumps(report,indent=2));groups.append(report)
        print(f'Prepared {count} slices, {report["spacing_mm"]:.4f} benchmark mm gap',flush=True)
    rng=np.random.default_rng(protocol['heldout_seed']);candidates=[]
    for i in range(protocol['common_heldout_planes']):
        for attempt in range(1000):
            height=float(low+(i+rng.uniform(.1,.9))/protocol['common_heldout_planes']*(high-low))
            if np.min(abs(np.asarray(all_heights)-height))>.03:break
        else:raise RuntimeError('Cannot construct shared disjoint heldout set')
        candidates.append(height)
    heldout=[dict(height_mm=h,rings_uv_mm=[r.tolist() for r in section_rings(mesh,h)]) for h in candidates]
    (ROOT/'data/heldout_contours.json').write_text(json.dumps(dict(planes=heldout),separators=(',',':')))
    boundaries=np.concatenate(all_bounds);lower=boundaries.min(0)-15;upper=boundaries.max(0)+15
    lower[2]=low;upper[2]=high
    domain=dict(lower_mm=lower.tolist(),upper_mm=upper.tolist(),definition='union of all prespecified observed boundary boxes plus 15 benchmark mm',
        common_heldout_count=len(heldout),minimum_distance_to_any_observed_mm=float(np.min(abs(np.asarray(candidates)[:,None]-np.asarray(all_heights)))),
        full_reference_mesh_used_for_loss=False,heldout_used_for_training_or_selection=False)
    (ROOT/'data/evaluation_domain.json').write_text(json.dumps(domain,indent=2))
    for count in protocol['slice_counts']:
        (ROOT/f'slices_{count:03d}/data/heldout_contours.json').symlink_to('../../data/heldout_contours.json')
    (ROOT/'data/preparation_report.json').write_text(json.dumps(dict(dataset='OReX Balloon Dog',
        original_stl_sha256=hashlib.sha256((ROOT/'source/balloondog_scaled.stl').read_bytes()).hexdigest(),
        vertices=len(mesh.vertices),faces=len(mesh.faces),watertight=bool(mesh.is_watertight),winding_consistent=bool(mesh.is_winding_consistent),
        original_source_bounds=original_bounds.tolist(),reference_bounds_mm=mesh.bounds.tolist(),
        coordinate_conversion=protocol['unit_convention'],rotation=np.eye(3).tolist(),reference_mesh_repaired=False,
        reference_connected_components=1,discarded_tip_extent_mm=.75,groups=groups,evaluation_domain=domain,
        multi_ring_labels='nearest true segment distance; even/odd sign over all rings; no connecting edges',
        reference_mesh_used_for_loss=False,heldout_used_for_training_or_selection=False),indent=2))


if __name__=='__main__':main()
