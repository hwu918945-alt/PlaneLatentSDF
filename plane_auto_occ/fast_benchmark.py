"""Measured A40 complete-case latency and original-versus-fast gradient gate."""
import argparse,copy,json,time
from pathlib import Path
import numpy as np
import torch
from .data import load_case
from .model import AutoOccupancy
from .population import VisitSampler
from .offline_pools import OfflineSampler,directory
from .offline_store import StaticSampler
from .chunked import backward_case as reference
from .fast_chunked import backward_case as fast
from .inference import bounds
from .area_pool import PlaneClassifier
from .fast_geometry import CompiledPlaneClassifier
from .cuda_geometry import CudaPlaneClassifier


def main():
    p=argparse.ArgumentParser();p.add_argument('--experiment',required=True);a=p.parse_args();root=Path(a.experiment);cfg=json.loads((root/'config.json').read_text())
    torch.set_num_threads(2);torch.manual_seed(cfg['seed']);device='cuda';c=load_case(root/'data/occupancy/train/002_ED.npz')
    sampler=OfflineSampler(directory(cfg['offline_pool_root'],c,0));sampler.set_visit(0,cfg['points'],cfg['seed']);c['_area_sampler']=StaticSampler(sampler,cfg['points'])
    for key in ('s','axis','boundary','boundary_endo'):c[key]=c[key].to(device)
    m=AutoOccupancy(cfg['latent_dim'],cfg['width']).to(device);z=torch.nn.Parameter(torch.randn(cfg['latent_dim'],device=device)*.01)
    initial=copy.deepcopy(m.state_dict());code=z.detach().clone();results=[];gradients=[];losses=[]
    for name,batch,fn in [('reference',4096,reference),('fast',32768,fast),('fast',65536,fast),('fast',131072,fast),('fast',524288,fast)]:
        m.load_state_dict(initial);z.data.copy_(code);configuration=dict(cfg,fast_query_points=batch);times=[]
        for repeat in range(4):
            m.zero_grad(set_to_none=True);z.grad=None;torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();start=time.perf_counter()
            loss,stats=fn(m,z,c,configuration,0);torch.cuda.synchronize();elapsed=time.perf_counter()-start
            if repeat:times.append(elapsed)
        g=torch.cat([v.grad.detach().flatten() for v in [z]+list(m.parameters())]);gradients.append(g.clone());losses.append(float(loss))
        results.append(dict(backend=name,query_batch=batch,mean_seconds=float(np.mean(times)),times=times,peak_cuda_bytes=torch.cuda.max_memory_allocated(),loss=float(loss),independent_points=stats['independent_points'],decoder_queries=stats['decoder_queries']))
    comparisons=[]
    for result,g,loss in zip(results[1:],gradients[1:],losses[1:]):
        error=(g-gradients[0]).norm()/gradients[0].norm();maxerror=(g-gradients[0]).abs().max();cosine=torch.nn.functional.cosine_similarity(g,gradients[0],dim=0)
        if float(error)>5e-5 or abs(loss-losses[0])>2e-6:raise AssertionError('Full-budget numerical equivalence failed')
        comparisons.append(dict(batch=result['query_batch'],relative_gradient_l2_error=float(error),max_gradient_error=float(maxerror),cosine=float(cosine),loss_absolute_error=abs(loss-losses[0])))
    cpu=load_case(root/'data/occupancy/train/310_ED.npz');lo,hi,basis=bounds(cpu,20);epi=cpu['boundary'][6].numpy()@basis[:,:2]*cpu['scale_mm'];endo=cpu['boundary_endo'][6].numpy()@basis[:,:2]*cpu['scale_mm']
    points=np.random.default_rng(7).uniform(epi.min(0)-20,epi.max(0)+20,(65536,2));geometry=[];expected=None
    for name,cls in [('reference_cpu',PlaneClassifier),('indexed_cpu',CompiledPlaneClassifier),('cuda_float64',CudaPlaneClassifier)]:
        classifier=cls(epi,endo);classifier.classify(points[:2]);start=time.perf_counter();answer=classifier.classify(points);seconds=time.perf_counter()-start
        if expected is None:expected=answer
        else:
            for x,y in zip(expected,answer):np.testing.assert_array_equal(x,y)
        geometry.append(dict(backend=name,seconds=seconds,all_labels_and_bands_equal=True))
    report=dict(device=torch.cuda.get_device_name(),torch=torch.__version__,architecture_and_objective_unchanged=True,amp=False,tf32=torch.backends.cuda.matmul.allow_tf32,complete_case=results,gradient_equivalence=comparisons,geometry=geometry)
    (root/'analysis/GPU_BENCHMARK.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
