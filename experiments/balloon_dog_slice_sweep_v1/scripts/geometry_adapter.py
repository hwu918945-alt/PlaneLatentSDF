"""Shared multi-ring geometry helpers from the validated Bunny adapter.

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

