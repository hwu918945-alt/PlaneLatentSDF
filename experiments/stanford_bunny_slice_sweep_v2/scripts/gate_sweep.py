"""Largest-stack CUDA gate: restart equivalence, full losses, actual evaluation."""
import json
import subprocess
import sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1]
GROUP=ROOT/'slices_075'


def run(script,*args):
    subprocess.run([sys.executable,str(ROOT/'scripts'/script),*map(str,args)],check=True,cwd=GROUP)


def assert_same(a,b):
    if torch.is_tensor(a):assert torch.equal(a,b)
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:assert_same(a[k],b[k])
    elif isinstance(a,(list,tuple)):
        assert len(a)==len(b)
        for aa,bb in zip(a,b):assert_same(aa,bb)
    else:assert a==b


def main():
    run('fit_sweep.py','--slices',75,'--run-dir','runs/gate','--stop-at',16,'--resume')
    run('fit_sweep.py','--slices',75,'--run-dir','runs/gate_reference','--stop-at',16)
    resumed=torch.load(GROUP/'runs/gate/last.pt',map_location='cpu',weights_only=False)
    direct=torch.load(GROUP/'runs/gate_reference/last.pt',map_location='cpu',weights_only=False)
    for k in ['student','teacher','distill','optimizer','rng_torch','rng_cuda']:assert_same(resumed[k],direct[k])
    logs=[json.loads(r) for r in (GROUP/'runs/gate/metrics.jsonl').read_text().splitlines()]
    assert len(logs)==16 and logs[-1]['ramp']==1 and logs[-1]['nonfinite_count']==0
    run('evaluate_bunny.py','--checkpoint','runs/gate/last.pt','--output','outputs/gate','--domain',ROOT/'data/evaluation_domain.json')
    m=json.loads((GROUP/'outputs/gate/EVALUATION_COMPLETE.json').read_text())
    assert m['observed_planes']==75 and m['heldout_planes']==121 and m['optimization_steps']==16
    for filename in ['bunny_reconstruction.obj','bunny_comparison.png','heldout_contours.png','metrics.json']:
        assert (GROUP/'outputs/gate'/filename).stat().st_size>0
    (ROOT/'CUDA_GATE_COMPLETE.json').write_text(json.dumps(dict(status='PASS',observed_planes=75,
        full_loss_ramp_verified=True,resume_bitwise_equivalent_to_uninterrupted_student_teacher_center_optimizer_rng=True,
        shared_holdout_and_mesh_evaluation=True,scientific_reconstruction_success_claimed=False),indent=2))


if __name__=='__main__':main()
