"""End-to-end GPU smoke throughput, including prefetch and complete Adam updates."""
import argparse,json,time
from pathlib import Path
import numpy as np,torch
from .population import fit
from .data import load_case
from .offline_pools import directory,prepare

def summarize(path,wall):
    rows=[json.loads(x) for x in (path/'history.jsonl').read_text().splitlines()]
    def mean(key):return float(np.mean([r[key] for r in rows]))
    return dict(updates=len(rows),wall_seconds=wall,mean_update_seconds=mean('update_seconds'),median_update_seconds=float(np.median([r['update_seconds'] for r in rows])),mean_data_wait_seconds=mean('data_wait_seconds'),mean_backward_seconds=mean('backward_seconds'),mean_optimizer_seconds=mean('optimizer_seconds'),mean_independent_points=mean('independent_points'),mean_decoder_queries=mean('decoder_queries'),max_cuda_bytes=max(r['peak_cuda_allocated_bytes'] for r in rows),checkpoint_step=torch.load(path/'last.pt',map_location='cpu',weights_only=False)['step'])

def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);a=p.parse_args();root=Path(a.experiment);cfg=json.loads((root/'config.json').read_text());lock=json.loads((root/'data_lock.json').read_text())
    ready=[]
    for row in lock['splits']['train']['records']:
        path=root/row['path'];c=load_case(path)
        if (directory(cfg['offline_pool_root'],c,0)/'q.npy').exists():ready.append(str(path))
        if len(ready)==16:break
    if len(ready)!=16:raise RuntimeError('Wait for 16 training pools')
    gate=root/'gates/throughput_pipeline';gate.mkdir(parents=True,exist_ok=True);joint=gate/'joint';configuration=dict(cfg,save_every=16)
    start=time.perf_counter();shared=fit(ready,[None]*16,configuration,joint,'cuda',steps=64);joint_report=summarize(joint,time.perf_counter()-start)
    row=lock['splits']['val']['records'][0];path=root/row['path'];observed=[0,row['planes']//2,row['planes']-1]
    # This explicit offline barrier completes before any frozen-code update.
    for generation in range(3):prepare(path,observed,cfg['offline_pool_root'],cfg,generation)
    frozen=gate/'frozen';start=time.perf_counter();fit([str(path)],[observed],configuration,frozen,'cuda',steps=200,pretrained=shared);frozen_report=summarize(frozen,time.perf_counter()-start)
    report=dict(joint=joint_report,frozen=frozen_report,train_phase_records=16,first_epoch=[r for r in ready],shared_freeze_verified=True,full_point_budget=True,scientific_success=False,interpretation='Small throughput pilot; final 570-case warmed epoch and validation/export overhead are not measured')
    (root/'analysis/PIPELINE_BENCHMARK.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
