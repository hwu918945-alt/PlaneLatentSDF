import torch
from torch import nn
from ..model import OccupancyDecoder


class CaseOnlyAutoOccupancy(nn.Module):
    def __init__(self,latent_dim=128,width=256):
        super().__init__();self.latent_dim=latent_dim;self.decoder=OccupancyDecoder(latent_dim,width)

    def plane_latents(self,code,s,phase):
        if code.numel()!=self.latent_dim:raise ValueError('Invalid case-code dimension')
        value=code if torch.is_grad_enabled() else code.detach()
        return value.reshape(1,-1).expand(len(s),-1)

    def field(self,q,s,u,axis,phase):
        h=q@axis
        if bool(((h.detach()<s[0]-1e-5)|(h.detach()>s[-1]+1e-5)).any()):raise ValueError('No axial extrapolation')
        return self.decoder(q,u[:1].expand(len(q),-1),phase)
