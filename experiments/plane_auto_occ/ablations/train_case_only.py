"""A only: original V2 sampler/chunked gradients, no Generator or LOO calls."""
import argparse,json,random,time
from pathlib import Path
import numpy as np
import torch
from ..area_pool import AreaSampler,build_pool,digest
from ..data import load_case
from ..chunked import backward_case
from .model_case_only import CaseOnlyAutoOccupancy


def initialize(cfg,case,device='cpu',step0=None):
    torch.set_num_threads(cfg['threads']);torch.manual_seed(cfg['seed']);np.random.seed(cfg['seed']);random.seed(cfg['seed'])
    model=CaseOnlyAutoOccupancy(cfg['latent_dim'],cfg['width']).to(device)
    codes=torch.nn.Parameter(torch.randn(1,cfg['latent_dim'],device=device)*.01)
    provenance=dict(kind='same-seed independent random initialization',decoder_code_exactly_paired=False,
        limitation='Removing G changes RNG consumption; decoder initial weights may differ from C')
    if step0:
        ck=torch.load(step0,map_location='cpu',weights_only=False)
        if ck['format']!='plane_auto_occ_v1' or ck['step']!=0:raise ValueError('Only C STEP ZERO is permitted for A initialization')
        if len(ck['records'])!=1 or ck['records'][0]['source_sha256']!=digest(case['path']) or ck['records'][0]['observed']!=case['observed']:
            raise ValueError('Step0 case/observations mismatch')
        if any(ck['config'][k]!=cfg[k] for k in ('latent_dim','width')):raise ValueError('Step0 architecture mismatch')
        state={k[len('decoder.'):]:v for k,v in ck['model'].items() if k.startswith('decoder.')}
        model.decoder.load_state_dict(state,strict=True)
        with torch.no_grad():codes.copy_(ck['codes'].to(device))
        if not all(torch.equal(v.cpu(),state[k].cpu()) for k,v in model.decoder.state_dict().items()):raise AssertionError('Decoder pairing failed')
        if not torch.equal(codes.detach().cpu(),ck['codes']):raise AssertionError('Code pairing failed')
        provenance=dict(kind='C step0 decoder and code ONLY',path=str(Path(step0).resolve()),sha256=digest(step0),
            source_step=0,decoder_code_exactly_paired=True,generator_copied=False,optimizer_copied=False)
    return model,codes,provenance


def train(case,observed,pool,cfg,output,device='cpu',steps=None,step0=None):
    cfg=dict(cfg)
    if steps is not None:cfg['steps']=steps
    if cfg.get('model_variant')!='case_only' or cfg['loo_mix']!=0 or cfg['loo_weight']!=0 or cfg['data_weight']!=1.5 or cfg['seam_weight']!=0:
        raise ValueError('A requires case_only, LOO0, data_weight1.5, Seam0')
    if cfg['steps']<1 or cfg['points']%cfg['microbatch_points'] or cfg['microbatch_points']%4:raise ValueError('Invalid update/point budget')
    out=Path(output)
    if out.exists() and any(out.iterdir()):raise FileExistsError('New/empty output required; no implicit resume')
    out.mkdir(parents=True,exist_ok=True);cpu=load_case(case,observed=observed)
    sampler=AreaSampler(pool,cfg['seed'])
    if sampler.meta['observed']!=cpu['observed'] or sampler.meta['source_sha256']!=digest(case):raise ValueError('Pool source/observations mismatch')
    model,codes,initialization=initialize(cfg,cpu,device,step0)
    opt=torch.optim.Adam([dict(params=[codes],lr=cfg['lr_latent']),dict(params=model.parameters(),lr=cfg['lr_shared'])])
    c={k:(v.to(device) if k in ('s','axis','boundary','boundary_endo') else v) for k,v in cpu.items()};c['_area_sampler']=sampler
    manifest=[dict(path=str(Path(case).resolve()),case_id=c['case_id'],phase=c['phase'],observed=c['observed'],source_sha256=digest(case))]
    metadata=dict(records=manifest,model_variant='case_only',mode='joint',config=cfg,initialization=initialization,
        loo='not_applicable: identical case code on every plane; no LOO decoder calls',
        decoder_parameters=sum(p.numel() for p in model.parameters()),code_parameters=codes.numel(),
        total_trainable_parameters=sum(p.numel() for p in model.parameters())+codes.numel(),
        initial_pool_sha256=digest(pool),initial_pool_array_sha256=sampler.meta['pool_sha256'])
    (out/'fit_manifest.json').write_text(json.dumps(metadata,indent=2));(out/'effective_config.json').write_text(json.dumps(cfg,indent=2))
    def save(step):
        payload=dict(format='plane_auto_occ_case_only_v1',model_variant='case_only',model=model.state_dict(),codes=codes.detach().cpu(),
            config=cfg,records=manifest,training_keys=[[c['case_id'],c['phase']]],mode='joint',step=step,initialization=initialization,
            sampling_pool=dict(path=sampler.path,sha256=digest(sampler.path),generation=sampler.meta['generation']),
            objective_version=cfg['objective_version'],loo='not_applicable')
        tmp=out/'last.pt.tmp';torch.save(payload,tmp);tmp.replace(out/'last.pt')
    save(0);initial=out/'step_00000';initial.mkdir();import shutil;shutil.copyfile(out/'last.pt',initial/'checkpoint.pt')
    order=[]
    with (out/'history.jsonl').open('w') as log:
        for step in range(cfg['steps']):
            start=time.perf_counter()
            if step and step%cfg['pool_refresh_updates']==0:
                generation=step//cfg['pool_refresh_updates'];target=out/'sampling_pools'/f'generation_{generation:03d}.npz'
                build_pool(case,observed,target,tuple(cfg['pool_capacities']),cfg['seed'],generation,cfg['pool_margin_mm'],cfg['boundary_band_mm'],cfg['near_mm'])
                sampler=AreaSampler(target,cfg['seed']);c['_area_sampler']=sampler
            if not order:order=torch.randperm(1).tolist()
            order.pop();opt.zero_grad(set_to_none=True)
            if device.startswith('cuda'):torch.cuda.reset_peak_memory_stats()
            loss,stats=backward_case(model,codes[0],c,cfg,step)
            if not bool(torch.isfinite(loss)):raise FloatingPointError('Nonfinite A loss')
            norm=torch.nn.utils.clip_grad_norm_([codes]+list(model.parameters()),cfg['clip_grad'],error_if_nonfinite=True)
            opt.step()
            if device.startswith('cuda'):torch.cuda.synchronize()
            row=dict(step=step+1,model_variant='case_only',case_id=c['case_id'],phase=c['phase'],loss=float(loss),gradient_norm=float(norm),**stats,
                loo_status='not_applicable',pool_generation=sampler.meta['generation'],update_seconds=time.perf_counter()-start,
                peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated() if device.startswith('cuda') else 0,
                peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved() if device.startswith('cuda') else 0)
            if row['decoder_queries']!=row['independent_points']:raise AssertionError('A accidentally evaluated LOO')
            log.write(json.dumps(row)+'\n');log.flush()
            if step==0 or (step+1)%cfg['log_every']==0:print(json.dumps(row),flush=True)
            if (step+1)%cfg['save_every']==0:save(step+1)
    save(cfg['steps'])
    (out/'TRAIN_COMPLETE.json').write_text(json.dumps(dict(steps=cfg['steps'],checkpoint_sha256=digest(out/'last.pt'),
        initialization=initialization,formal_fixed_final_checkpoint=True),indent=2))
    return out/'last.pt'


def main():
    p=argparse.ArgumentParser();p.add_argument('--case',required=True);p.add_argument('--observed',default='0,3,6');p.add_argument('--area-pool',required=True)
    p.add_argument('--config',default=str(Path(__file__).with_name('config_A_case_only.json')));p.add_argument('--output',required=True)
    p.add_argument('--device',default='cpu');p.add_argument('--steps',type=int);p.add_argument('--init-from-c-step0')
    a=p.parse_args();train(a.case,[int(x) for x in a.observed.split(',')],a.area_pool,json.loads(Path(a.config).read_text()),a.output,a.device,a.steps,a.init_from_c_step0)

if __name__=='__main__':main()
