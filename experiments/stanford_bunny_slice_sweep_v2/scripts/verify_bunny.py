import json
import sys
from pathlib import Path
import numpy as np
import torch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'runtime'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
from prepare_bunny import region_sdf
from core.data import load_case,gap_queries
from core.model import PlaneField,make_teacher
from core.losses import DistanceWeightedDistillation,eikonal_loss
from core.initialization import initialize_decoder
from core.training import case_objective

square=np.array([[-1,-1],[1,-1],[1,1],[-1,1]],float)
points=np.array([[0,0],[.75,0],[1,0],[2,0]],float)
d=region_sdf(points,[square,square*.5]);assert np.allclose(d,[.5,-.25,0,1])
disjoint=region_sdf(np.array([[0,0],[3,0],[1.5,0]],float),[square,square+[3,0]])
assert np.allclose(disjoint,[-1,-1,.5])
c=load_case('data/bunny_observed.npz');planes=json.loads(Path('data/observed_contours.json').read_text())['planes']
max_error=0
for i,p in enumerate(planes):
    world=c['boundary'][i].numpy()*100+c['center_mm']
    error=float(abs(region_sdf(world[:,:2],p['rings_uv_mm'])).max());max_error=max(max_error,error)
    assert error<1e-3
    xy=c['q'][i].numpy()[:,:2]*100+c['center_mm'][:2]
    actual=region_sdf(xy,p['rings_uv_mm'])/100
    assert np.max(abs(actual-c['sdf2d'][i].numpy()))<1e-5
torch.set_num_threads(1);torch.manual_seed(20261009)
cfg=json.loads(Path('config.json').read_text());cfg.update(warmup_epochs=0,ramp_epochs=1)
m=PlaneField(**cfg['model']);initialize_decoder(m);t=make_teacher(m);distill=DistanceWeightedDistillation()
loss,logs,_=case_objective(m,t,distill,c,cfg,0);loss.backward()
assert all(p.grad is None for p in t.parameters())
gradients={}
for name in ['encoder','decoder','head']:
    gradients[name]=sum(float(p.grad.abs().sum()) for p in getattr(m,name).parameters() if p.grad is not None)
    assert gradients[name]>0
m.zero_grad();q,_,_=gap_queries(c,32);z=m.encoder(c['q'],c['sdf2d'],c['phase'])
eikonal_loss(lambda x:m.field(x,c['s'],z,c['axis'],c['phase']),q).backward()
assert m.encoder.point[0].weight.grad.abs().sum()>0
report=dict(status='PASS',multi_ring_evenodd_sign_and_segment_distance=True,max_boundary_distance_mm=max_error,
            full_loss_backward_gradients=gradients,teacher_frozen=True,eikonal_independent_encoder_gradient=True,losses=logs)
Path('data/verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
