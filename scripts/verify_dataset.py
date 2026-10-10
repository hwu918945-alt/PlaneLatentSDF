import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from core.data import load_case,validate_split,polygon_sdf
from core.mesh import stack_basis


root=Path('data/plane_sdf')
paths={k:sorted((root/k).glob('*.npz')) for k in ['train','val','test']}
validate_split(paths)
manifest=json.loads(Path('data/aligned_sax_manifest.json').read_text())
records={(r['case_id'],0 if r['phase']=='ED' else 1):r for r in manifest['records']}
report={'patients':{},'phases':{},'max_plane_residual_mm':0.,'max_boundary_distance_mm':0.,'min_gap_mm':1e6,'records':[]}
for split,files in paths.items():
    ids=set()
    for path in files:
        c=load_case(path);ids.add(c['case_id'])
        assert c['scale_mm']==100.
        q,d,b,s,axis=[c[k].numpy() for k in ['q','sdf2d','boundary','s','axis']]
        assert len(s)>=3 and np.any(d<0) and np.any(d>0)
        residual=float(abs(q@axis-s[:,None]).max()*100)
        report['max_plane_residual_mm']=max(report['max_plane_residual_mm'],residual)
        report['min_gap_mm']=min(report['min_gap_mm'],float(np.diff(s).min()*100))
        record=records[(c['case_id'],c['phase'])]
        planes=sorted(record['planes'],key=lambda p:np.asarray(p['origin_mm'])@axis)
        for i,p in enumerate(planes):
            world=b[i]*100+c['center_mm']
            delta=world-np.array(p['origin_mm'])
            uv=np.stack([delta@p['u'],delta@p['v']],-1)
            error=float(abs(polygon_sdf(uv,p['contour_uv_mm'])).max())
            report['max_boundary_distance_mm']=max(report['max_boundary_distance_mm'],error)
            assert error<1e-3,(path,error)
        report['records'].append(dict(file=str(path),planes=len(s),inside_fraction=float((d<0).mean()),outside_fraction=float((d>0).mean())))
    report['patients'][split]=len(ids);report['phases'][split]=len(files)
assert report['patients']=={'train':285,'val':36,'test':36}
assert report['phases']=={'train':570,'val':72,'test':72}
out=Path('preflight/dataset');out.mkdir(parents=True,exist_ok=True)
for path in paths['train'][:2]+paths['val'][:2]:
    c=load_case(path);basis=stack_basis(c['axis'].numpy())
    chosen=[0,len(c['s'])//2,len(c['s'])-1]
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,i in zip(axes,chosen):
        q=c['q'][i].numpy()@basis[:,:2]*100;b=c['boundary'][i].numpy()@basis[:,:2]*100
        ax.scatter(q[:,0],q[:,1],c=c['sdf2d'][i].numpy()*100,s=3,cmap='coolwarm',vmin=-6,vmax=6)
        ax.plot(*np.r_[b,b[:1]].T,'k-',linewidth=1);ax.set_aspect('equal');ax.set_title(f'plane {i}, signed distance mm')
    fig.suptitle(path.stem);fig.tight_layout();fig.savefig(out/f'{path.stem}_sdf.png',dpi=150);plt.close(fig)
    fig=plt.figure(figsize=(6,6));ax=fig.add_subplot(projection='3d')
    for b in c['boundary'].numpy()*100+c['center_mm']:ax.plot(*b.T)
    ax.set_box_aspect(np.ptp(c['boundary'].numpy().reshape(-1,3),axis=0));fig.savefig(out/f'{path.stem}_stack.png',dpi=150);plt.close(fig)
# Local symlinks only, retaining the patient-level fixed train/val split.
for name,train_count in [('real_smoke',4),('single_patient',2)]:
    for split,entries in [('train',paths['train'][:train_count]),('val',paths['val'][:2])]:
        folder=Path('preflight')/name/split;folder.mkdir(parents=True,exist_ok=True)
        for path in entries:
            target=folder/path.name
            if not target.exists():target.symlink_to(path.resolve())
report['status']='PASS'
(out/'dataset_audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({k:v for k,v in report.items() if k!='records'},indent=2))
