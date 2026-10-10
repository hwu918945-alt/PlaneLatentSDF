"""Public Bunny -> closed benchmark reference -> sparse parallel multi-ring input.

Only sampled observed sections produce training labels. Hole patches are an
explicit reference-mesh preparation assumption, never predicted output caps.
"""
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path
import numpy as np
import trimesh
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
from core.data import polygon_sdf,resample_contour,load_case


def cross(a,b):
    return a[...,0]*b[...,1]-a[...,1]*b[...,0]


def triangulate_hole(vertices,loop):
    p=vertices[loop];_,_,vh=np.linalg.svd(p-p.mean(0),full_matrices=False)
    uv=(p-p.mean(0))@vh[:2].T
    order=list(range(len(loop)))
    area=cross(uv,np.roll(uv,-1,axis=0)).sum()/2
    if area<0:order=order[::-1]
    faces=[]
    while len(order)>3:
        found=False
        for pos in range(len(order)):
            a,b,c=order[pos-1],order[pos],order[(pos+1)%len(order)]
            if cross(uv[b]-uv[a],uv[c]-uv[b])<=1e-10:continue
            other=[j for j in order if j not in [a,b,c]]
            t=uv[other]
            inside=(cross(uv[b]-uv[a],t-uv[a])>=-1e-10)&(cross(uv[c]-uv[b],t-uv[b])>=-1e-10)&(cross(uv[a]-uv[c],t-uv[c])>=-1e-10)
            if inside.any():continue
            faces.append([loop[a],loop[b],loop[c]]);order.pop(pos);found=True;break
        if not found:raise ValueError('Non-simple hole projection: refuse guessed triangulation')
    faces.append([loop[j] for j in order])
    return faces,dict(boundary_vertices=len(loop),pca_plane_max_residual_mm=float(abs((p-p.mean(0))@vh[2]).max()),projected_area_mm2=float(abs(area)))


def repair_reference(mesh):
    edges=mesh.edges
    _,inverse,count=np.unique(np.sort(edges,axis=1),axis=0,return_inverse=True,return_counts=True)
    boundary=edges[count[inverse]==1]
    adjacency=defaultdict(list)
    for a,b in boundary:
        adjacency[int(a)].append(int(b));adjacency[int(b)].append(int(a))
    if not all(len(v)==2 for v in adjacency.values()):raise ValueError('Nonmanifold input boundary')
    left=set(adjacency);new_faces=[];report=[]
    while left:
        first=min(left);loop=[first];prev=-1;current=first
        while True:
            following=next(x for x in adjacency[current] if x!=prev)
            if following==first:break
            if following in loop:raise ValueError('Broken boundary cycle')
            loop.append(following);prev,current=current,following
        left.difference_update(loop)
        faces,details=triangulate_hole(mesh.vertices,loop);new_faces.extend(faces);report.append(details)
    repaired=trimesh.Trimesh(vertices=mesh.vertices.copy(),faces=np.r_[mesh.faces,np.asarray(new_faces)],process=False)
    trimesh.repair.fix_normals(repaired,multibody=True)
    if not repaired.is_watertight or not repaired.is_winding_consistent:raise ValueError('Repair failed topology QC')
    return repaired,report


def section_rings(mesh,z):
    section=mesh.section(plane_normal=[0,0,1],plane_origin=[0,0,float(z)])
    if section is None:return []
    rings=[]
    for curve in section.discrete:
        if np.linalg.norm(curve[0]-curve[-1])>1e-4:raise ValueError(f'Open reference section at {z}')
        uv=curve[:-1,:2]
        uv=uv[np.linalg.norm(uv-np.roll(uv,1,axis=0),axis=1)>1e-7]
        if len(uv)<3:raise ValueError('Degenerate section')
        area=cross(uv,np.roll(uv,-1,axis=0)).sum()/2
        if area<0:uv=uv[::-1]
        centered=uv-uv.mean(0);start=np.argmin(abs(np.arctan2(centered[:,1],centered[:,0])))
        rings.append(np.roll(uv,-int(start),axis=0))
    return sorted(rings,key=lambda r:tuple(r.mean(0)))


def region_sdf(points,rings):
    """Nearest segment distance with even/odd interior across ALL closed rings."""
    values=np.stack([polygon_sdf(points,ring) for ring in rings])
    inside=(values<0).sum(0)%2==1
    return np.where(inside,-np.min(abs(values),axis=0),np.min(abs(values),axis=0)).astype(np.float32)


def all_boundary(rings,count):
    lengths=np.asarray([np.linalg.norm(np.roll(r,-1,axis=0)-r,axis=1).sum() for r in rings])
    counts=np.full(len(rings),8)
    extra=(count-counts.sum())*lengths/lengths.sum()
    counts+=np.floor(extra).astype(int)
    counts[np.argsort(extra-np.floor(extra))[:-(count-counts.sum())-1:-1]]+=1 if count-counts.sum() else 0
    if counts.sum()!=count:raise ValueError('Boundary allocation')
    return np.concatenate([resample_contour(r,int(n)) for r,n in zip(rings,counts)])


def main():
    root=Path('.')
    mesh=trimesh.load('source/bun_zipper.ply',process=True)
    # Explicit benchmark unit convention, not a claim of verified scan units.
    mesh.vertices*=1000.
    rotation=np.array([[1.,0,0],[0,0,-1],[0,1,0]])
    mesh.vertices=mesh.vertices@rotation.T
    mesh.export('source/bunny_original_mm.ply')
    gt,holes=repair_reference(mesh);gt.export('source/bunny_reference_mm.ply');gt.export('source/bunny_reference_mm.obj')
    low,high=gt.bounds[:,2];heights=np.linspace(low+.75,high-.75,16)
    rings=[section_rings(gt,h) for h in heights]
    boundary=[all_boundary(r,256) for r in rings]
    world_b=np.asarray([np.c_[b,np.full(len(b),h)] for b,h in zip(boundary,heights)])
    center=world_b.reshape(-1,3).mean(0)
    q,d=[],[];rng=np.random.default_rng(20261009)
    for b,h,r in zip(boundary,heights,rings):
        seeds=b[rng.integers(len(b),size=768)]
        sigma=rng.choice([1.,3.,6.],size=768,p=[.4,.4,.2])
        near=seeds+rng.normal(size=seeds.shape)*sigma[:,None]
        points=np.r_[near,rng.uniform(b.min(0)-10,b.max(0)+10,(256,2))]
        q.append((np.c_[points,np.full(len(points),h)]-center)/100)
        d.append(region_sdf(points,r)/100)
    np.savez_compressed('data/bunny_observed.npz',q=np.array(q,np.float32),sdf2d=np.array(d,np.float32),
                        boundary=np.array((world_b-center)/100,np.float32),s=np.array((heights-center[2])/100,np.float32),
                        axis=np.array([0,0,1],np.float32),center_mm=center,scale_mm=np.array(100.),phase=np.array(0),case_id=np.array('stanford_bunny'))
    load_case('data/bunny_observed.npz')
    observations=dict(heights_mm=heights.tolist(),planes=[dict(height_mm=float(h),rings_uv_mm=[r.tolist() for r in rr]) for h,rr in zip(heights,rings)])
    Path('data/observed_contours.json').write_text(json.dumps(observations,separators=(',',':')))
    heldout=[]
    for a,b in zip(heights[:-1],heights[1:]):
        for fraction in [.25,.5,.75]:
            z=float(a+fraction*(b-a));rr=section_rings(gt,z)
            heldout.append(dict(height_mm=z,rings_uv_mm=[r.tolist() for r in rr]))
    Path('data/heldout_contours.json').write_text(json.dumps(dict(planes=heldout),separators=(',',':')))
    report=dict(dataset='Stanford Bunny',source_url='https://graphics.stanford.edu/pub/3Dscanrep/bunny.tar.gz',
                attribution='Stanford University Computer Graphics Laboratory',archive_sha256=hashlib.sha256(Path('source/bunny.tar.gz').read_bytes()).hexdigest(),
                unit_convention='1 source coordinate unit = 1000 benchmark mm; source units not independently verified',
                rotation=rotation.tolist(),original_vertices=len(mesh.vertices),original_faces=len(mesh.faces),original_watertight=bool(mesh.is_watertight),
                reference_vertices=len(gt.vertices),reference_faces=len(gt.faces),reference_watertight=bool(gt.is_watertight),
                explicitly_repaired_holes=holes,repair='PCA-plane ear clipping, original boundary vertices preserved; reference geometry assumption',
                reference_bounds_mm=gt.bounds.tolist(),observed_planes=16,heldout_planes=len(heldout),observed_ring_counts=[len(r) for r in rings],
                multi_ring_labels='minimum true segment distance; even/odd sign across all loops, no invented connecting edges',
                scale_mm=100.,center_from_observed_contours_mm=center.tolist(),observed_extent_mm=[float(heights[0]),float(heights[-1])],
                discarded_tip_extent_mm=.75,full_mesh_used_for_training_loss=False,heldout_used_for_training_or_selection=False)
    Path('data/preparation_report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
