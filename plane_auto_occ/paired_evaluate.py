"""Fixed native-grid evaluation; original hidden subset is now observed fitting."""
import json
from pathlib import Path
import numpy as np
from .area_pool import digest
from .inference import restore,bounds,plane_grid,query
from .protocol import evaluate as upstream_evaluate,labels


def evaluate(checkpoint,case,output,device='cpu',resolution=160,mesh_resolution=96):
    out=Path(output);report=upstream_evaluate(checkpoint,case,out,device,resolution,mesh_resolution)
    model,code,u,obs,full=restore(checkpoint,case,device);lower,upper,basis=bounds(obs,20.)
    reference=set(obs['_roi_reference_observed']);rows=[]
    for i,pos in enumerate(full['s'].numpy()):
        if not lower[2]-1e-6<=pos<=upper[2]+1e-6:continue
        q=plane_grid(pos,lower,upper,basis,resolution);logits=query(model,q,obs,u);pred=logits>=0;gt=labels(full,i,q,basis)
        tp=int((pred&gt).sum());ps=int(pred.sum());gs=int(gt.sum())
        probability=1/(1+np.exp(-np.clip(logits,-80,80)))
        rows.append(dict(index=i,original_subset='original_three' if i in reference else 'original_hidden_now_observed',
            dice=float((2*tp+1e-6)/(ps+gs+1e-6)),precision=tp/ps if ps else 0.,recall=tp/gs if gs else None,
            predicted_gt_area_ratio=ps/gs if gs else None,gt_positive_pixels=gs,predicted_positive_pixels=ps,
            probability_std=float(probability.std()),logit_std=float(logits.std()),empty_prediction=ps==0,
            full_prediction=ps==len(pred)))
    oldhidden=[r['dice'] for r in rows if r['original_subset']=='original_hidden_now_observed']
    report.update(fixed_grid_planes=rows,resolution=resolution,mesh_resolution=mesh_resolution,roi_margin_mm=20.,
        roi_reference_observed=sorted(reference),all_sax_mean_dice=float(np.mean([r['dice'] for r in rows])),
        original_hidden_subset_fit_mean_dice=float(np.mean(oldhidden)) if oldhidden else None,
        original_hidden_subset_is_independent_test=False,checkpoint_sha256=digest(checkpoint))
    (out/'metrics.json').write_text(json.dumps(report,indent=2));return report
