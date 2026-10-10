"""Generate shell fixtures for engineering smoke tests, not cardiac accuracy claims."""
import argparse
import json
from pathlib import Path
import numpy as np
from prepare_cache import convert


def record(cid, split, phase, seed):
    rng=np.random.default_rng(seed)
    angle=np.arange(96)/96*2*np.pi
    planes=[]
    for i,h in enumerate((0.,7.,18.,31.,42.)):
        outer=25.+2*np.sin(i*.7)+rng.uniform(-.5,.5);inner=outer-6.
        if phase=='ES':outer*=.9;inner*=.8
        def polygon(r):return np.stack((r*np.cos(angle),.8*r*np.sin(angle)),-1).tolist()
        planes.append(dict(origin_mm=[12.,-9.,40.+h],u=[1.,0.,0.],v=[0.,1.,0.],
                           epi_uv_mm=polygon(outer),endo_uv_mm=polygon(inner),source_slice_index=i))
    return dict(case_id=cid,split=split,phase=phase,anatomy='LV_myocardium_shell',planes=planes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--output',default='preflight/synthetic_shell');args=p.parse_args()
    root=Path(args.output);records=[]
    for split in ('train','val','test'):
        for phase in ('ED','ES'):
            r=record('fixture_'+split,split,phase,7);records.append(r)
            convert(r,root/split/f'{r["case_id"]}_{phase}.npz',np.random.default_rng(42))
    (root/'manifest.json').write_text(json.dumps(dict(records=records)))
    print(root)


if __name__=='__main__':main()
