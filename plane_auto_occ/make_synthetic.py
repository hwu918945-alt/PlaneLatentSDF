"""Nonclinical elliptical myocardium shell fixtures, in baseline cache format."""
import argparse
from pathlib import Path
import numpy as np
from .geometry import stack_basis,contour_labels


def make_case(path,case_id='synthetic_0',seed=19,phase=0):
    rng=np.random.default_rng(seed); s=np.linspace(-.3,.3,7).astype(np.float32)
    angle=np.linspace(0,2*np.pi,128,endpoint=False)
    qs,ds,epis,endos=[],[],[],[]; axis=np.array([0,0,1],np.float32)
    for pos in s:
        shrink=1-.6*(pos/.6)**2; ax=.35*shrink; ay=.29*shrink
        center=np.array([.04*pos,.025*pos])
        epi=np.c_[ax*np.cos(angle),ay*np.sin(angle)]+center
        endo=np.c_[.68*ax*np.cos(angle),.68*ay*np.sin(angle)]+center
        epi3=np.c_[epi,np.full(len(epi),pos)].astype(np.float32)
        endo3=np.c_[endo,np.full(len(endo),pos)].astype(np.float32)
        uv=rng.uniform(-.5,.5,(4096,2)); q=np.c_[uv,np.full(len(uv),pos)].astype(np.float32)
        _,de,di=contour_labels(q,epi3,endo3,stack_basis(axis))
        qs.append(q); ds.append(np.maximum(de,-di)); epis.append(epi3); endos.append(endo3)
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,q=np.array(qs),sdf2d=np.array(ds),boundary=np.array(epis),boundary_endo=np.array(endos),
                        s=s,axis=axis,center_mm=np.zeros(3),scale_mm=np.array(100.),phase=np.array(phase),case_id=np.array(case_id))


def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
    for split in ('train','test'):
        make_case(Path(a.output)/split/f'{split}_ED.npz',case_id=f'synthetic_{split}',seed=19 if split=='train' else 20)
    print('Synthetic fixtures only; no clinical metrics')


if __name__ == '__main__': main()
