"""Audit shell pools against continuous source polygons and fixed patient splits."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from core.data import load_case, require_shell, validate_split
from core.geometry import shell_sdf
from core.mesh import stack_basis
from core.sampling import SOURCE_NAMES, POOL_COUNTS


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data/plane_sdf_shell'))
    p.add_argument('--manifest',type=Path,default=Path('data/aligned_sax_shell_manifest.json'))
    p.add_argument('--output',type=Path,default=Path('preflight/shell_dataset_audit.json'))
    p.add_argument('--allow-subset',action='store_true',help='Engineering fixtures only; skips full 285/36/36 counts')
    args=p.parse_args()
    paths={k:sorted((args.data/k).glob('*.npz')) for k in ('train','val','test')}
    if not any(paths.values()):raise ValueError('No shell caches found')
    validate_split(paths)
    records={(r['case_id'],0 if r['phase']=='ED' else 1):r for r in json.loads(args.manifest.read_text())['records']}
    report=dict(status='PASS',patients={},phase_files={},max_plane_residual_mm=0.,max_sdf_error_mm=0.,
                source_counts=dict(zip(SOURCE_NAMES,POOL_COUNTS)),subset_only=args.allow_subset,records=[])
    seen=set()
    for split,files in paths.items():
        ids={};phases={}
        for path in files:
            c=load_case(path);require_shell(c)
            key=(c['case_id'],c['phase']);seen.add(key)
            if key not in records or records[key]['split']!=split:raise ValueError(f'Manifest/split mismatch: {path}')
            if c['phase'] in phases.setdefault(c['case_id'],set()):raise ValueError('Duplicate phase')
            phases[c['case_id']].add(c['phase']);ids[c['case_id']]=1
            q,d,b,s,axis=[c[k].numpy() for k in ('q','sdf2d','boundary','s','axis')]
            basis=stack_basis(axis);plane_map={p['source_slice_index']:p for p in records[key]['planes']}
            for i,source_id in enumerate(c['source_slice_indices']):
                plane=plane_map[int(source_id)]
                origin=np.array(plane['origin_mm']);u=np.array(plane['u']);v=np.array(plane['v'])
                polygons=[]
                for name in ('epi_uv_mm','endo_uv_mm'):
                    uv=np.array(plane[name]);polygons.append((uv[:,:1]*u+uv[:,1:]*v)@basis[:,:2])
                world=q[i]*100+c['center_mm'];query=(world-origin)@basis[:,:2]
                expected=shell_sdf(query,*polygons)
                error=float(np.max(abs(expected-d[i]*100)))
                report['max_sdf_error_mm']=max(report['max_sdf_error_mm'],error)
                if error>.002:raise ValueError(f'Continuous source SDF mismatch: {path}/{i}: {error}')
            residual=float(abs(q@axis-s[:,None]).max()*100)
            report['max_plane_residual_mm']=max(report['max_plane_residual_mm'],residual)
            report['records'].append(dict(file=str(path),planes=len(s),case_id=c['case_id'],phase=c['phase']))
        if phases and any(pair!={0,1} for pair in phases.values()):raise ValueError(f'Missing ED/ES pair: {split}')
        report['patients'][split]=len(ids);report['phase_files'][split]=len(files)
    if seen!=set(records):raise ValueError('Prepared files do not cover source manifest exactly')
    if not args.allow_subset and report['patients']!={'train':285,'val':36,'test':36}:
        raise ValueError(f'Fixed patient count mismatch: {report["patients"]}')
    args.output.parent.mkdir(parents=True,exist_ok=True);args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))


if __name__=='__main__':main()
