"""Generate observations and one disjoint, shared evaluation plane set."""
import hashlib
import json
import shutil
import sys
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from core.data import load_case
from prepare_bunny import section_rings,all_boundary,region_sdf


def main():
    root=Path(__file__).resolve().parents[1]
    protocol=json.loads((root/'protocol.json').read_text())
    reference=trimesh.load(root/'source/bunny_reference_mm.ply',process=False)
    low,high=reference.bounds[:,2]+np.array([.75,-.75])
    all_heights=[];all_bounds=[]
    for count in protocol['slice_counts']:
        run=root/f'slices_{count:03d}'
        if count==16:
            prior=root.parent/'stanford_bunny_single_case_v1/data'
            for name in ['bunny_observed.npz','observed_contours.json']:
                shutil.copy2(prior/name,run/'data'/name)
            planes=json.loads((run/'data/observed_contours.json').read_text())['planes']
            heights=np.asarray([p['height_mm'] for p in planes])
        else:
            heights=np.linspace(low,high,count)
            rings=[section_rings(reference,h) for h in heights]
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
            np.savez_compressed(run/'data/bunny_observed.npz',q=np.asarray(q,np.float32),sdf2d=np.asarray(d,np.float32),
                boundary=np.asarray((world-center)/100,np.float32),s=np.asarray((heights-center[2])/100,np.float32),
                axis=np.array([0,0,1],np.float32),center_mm=center,scale_mm=np.array(100.),phase=np.array(0),case_id=np.array('stanford_bunny'))
            planes=[dict(height_mm=float(h),rings_uv_mm=[r.tolist() for r in rr]) for h,rr in zip(heights,rings)]
            (run/'data/observed_contours.json').write_text(json.dumps(dict(planes=planes),separators=(',',':')))
        case=load_case(run/'data/bunny_observed.npz')
        assert len(case['s'])==count
        assert float((case['s'][1:]-case['s'][:-1]).min())>4*.003
        world=case['boundary'].numpy()*100+case['center_mm']
        all_bounds.append(world.reshape(-1,3));all_heights.extend(heights.tolist())
        (run/'data/preparation_report.json').write_text(json.dumps(dict(observed_planes=count,extent_mm=[float(heights[0]),float(heights[-1])],
            spacing_mm=float(np.diff(heights).mean()),scale_mm=100,center_mm=case['center_mm'].tolist(),
            boundary_ring_counts=[len(p['rings_uv_mm']) for p in planes],reference_mesh_used_for_loss=False),indent=2))
        print(f'Prepared {count} observed slices',flush=True)
    # One jittered, approximately uniform set, at least 0.03 mm from ANY group.
    rng=np.random.default_rng(protocol['heldout_seed']);candidates=[]
    for i in range(protocol['common_heldout_planes']):
        for attempt in range(1000):
            fraction=(i+rng.uniform(.1,.9))/protocol['common_heldout_planes']
            height=float(low+fraction*(high-low))
            if np.min(abs(np.asarray(all_heights)-height))>.03:break
        else:raise RuntimeError('Cannot generate disjoint heldout plane')
        candidates.append(height)
    heldout=[dict(height_mm=h,rings_uv_mm=[r.tolist() for r in section_rings(reference,h)]) for h in candidates]
    (root/'data/heldout_contours.json').write_text(json.dumps(dict(planes=heldout),separators=(',',':')))
    bounds=np.concatenate(all_bounds);lower=bounds.min(0)-15;upper=bounds.max(0)+15
    lower[2]=low;upper[2]=high
    (root/'data/evaluation_domain.json').write_text(json.dumps(dict(lower_mm=lower.tolist(),upper_mm=upper.tolist(),
        definition='union of observed boundary boxes across prespecified groups; no full-reference bbox',
        common_heldout_count=len(heldout),minimum_distance_to_any_observed_mm=float(np.min(abs(np.asarray(candidates)[:,None]-np.asarray(all_heights)))),
        full_reference_mesh_used_for_loss=False,heldout_used_for_training_or_selection=False),indent=2))
    for count in protocol['slice_counts']:
        target=root/f'slices_{count:03d}/data/heldout_contours.json'
        target.symlink_to('../../data/heldout_contours.json')
    hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for p in root.rglob('*.npz')}
    hashes['data/heldout_contours.json']=hashlib.sha256((root/'data/heldout_contours.json').read_bytes()).hexdigest()
    (root/'data/sha256.json').write_text(json.dumps(hashes,indent=2))


if __name__=='__main__':main()
