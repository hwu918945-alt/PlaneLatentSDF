import argparse
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser(); p.add_argument('--root',required=True); p.add_argument('--expected-patients',default='285,36,36')
    a=p.parse_args(); expected=[int(x) for x in a.expected_patients.split(',')]
    if len(expected) != 3: raise ValueError('Need train,val,test counts')
    sets={}; report={}
    for split,count in zip(('train','val','test'),expected):
        keys=[]
        for path in sorted((Path(a.root)/split).glob('*.npz')):
            with np.load(path,allow_pickle=False) as f: keys.append((str(f['case_id']),int(f['phase'])))
        if len(keys) != len(set(keys)): raise ValueError(f'{split}: duplicate patient/phase')
        patients={k[0] for k in keys}; sets[split]=patients
        if len(patients) != count: raise ValueError(f'{split}: {len(patients)} patients, expected {count}')
        if set(keys) != {(c,p) for c in patients for p in (0,1)}:
            raise ValueError(f'{split}: need ED and ES for every patient')
        report[split]=dict(patients=len(patients),phase_files=len(keys))
    for s,t in (('train','val'),('train','test'),('val','test')):
        if sets[s] & sets[t]: raise ValueError(f'Patient leakage: {s}/{t}')
    print(json.dumps(report,indent=2))


if __name__ == '__main__': main()
