"""Uniform float64 evaluation adapter; no new model parameters or field math.

Native query() creates float32 xyz tensors. Promote these same rounded xyz to
float64 before the unchanged model.field/interpolate/decoder calls, to avoid
float32 (1-alpha)+alpha rounding at oblique observed planes. G's saved float32
outputs are computed once by native restore and promoted identically for B/C.
Training is untouched. Raw float32 BC discrepancies remain in the audit.
"""
import torch


class EvaluationPrecision:
    def __init__(self,model):
        self.model=model.double()
    @property
    def decoder(self):return self.model.decoder
    def state_dict(self):return self.model.state_dict()
    def field(self,q,s,u,axis,phase):
        return self.model.field(q.to(torch.float64),s,u,axis,phase)


def promote(item):
    model,code,u,obs,full=item;obs=dict(obs)
    obs['s']=obs['s'].double();obs['axis']=obs['axis'].double()
    return EvaluationPrecision(model),code.double(),u.double(),obs,full
