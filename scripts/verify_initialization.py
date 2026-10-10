import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from core.model import PlaneField,make_teacher
from core.initialization import initialize_decoder
from core.data import load_case,gap_queries
from core.losses import DistanceWeightedDistillation,eikonal_loss
from core.training import case_objective

torch.set_num_threads(1);torch.manual_seed(20261009)
cfg=json.loads(Path('config.json').read_text());cfg.update(warmup_epochs=0,ramp_epochs=1)
m=PlaneField(**cfg['model']);report=initialize_decoder(m);teacher=make_teacher(m)
q=torch.randn(48,3)*.2;z=torch.zeros(48,cfg['model']['latent_dim'])
with torch.no_grad():
    symmetry=float((m.decoder(q,z,0)-m.decoder(-q,z,0)).abs().max())
    assert symmetry<1e-5
    assert m.decoder(torch.zeros(1,3),z[:1],0).item()<0
    probes=torch.eye(3)*.6
    assert (m.decoder(probes,z[:3],0)>0).all()
c=load_case('preflight/single_patient/train/002_ED.npz')
gap,_,_=gap_queries(c,32);latent=m.encoder(c['q'],c['sdf2d'],c['phase'])
eik=eikonal_loss(lambda x:m.field(x,c['s'],latent,c['axis'],c['phase']),gap)
eik.backward();encoder_gradient=float(m.encoder.point[0].weight.grad.abs().sum());assert encoder_gradient>0
m.zero_grad();loss,logs,_=case_objective(m,teacher,DistanceWeightedDistillation(),c,cfg,0);loss.backward()
assert logs['calibration_valid_fraction']>.1
for group in [m.encoder,m.decoder,m.head]:
    grads=[p.grad for p in group.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads) and sum(float(g.abs().sum()) for g in grads)>0
assert all(p.grad is None for p in teacher.parameters())
report.update(status='PASS',symmetry_error=symmetry,eikonal_only_encoder_gradient=encoder_gradient,full_losses=logs)
Path('preflight/initialization_verification.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
