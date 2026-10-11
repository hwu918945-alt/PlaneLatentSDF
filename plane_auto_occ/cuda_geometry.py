"""Optional exact float64 CUDA classifier; same proposal RNG and band contract."""
import numpy as np
import torch
from .area_pool import PlaneClassifier

class CudaPlaneClassifier(PlaneClassifier):
    def __init__(self,epi,endo):
        super().__init__(epi,endo)
        self.polygons=[torch.as_tensor(e[0],dtype=torch.float64,device='cuda') for e in self.edges]
    @torch.no_grad()
    def classify(self,points,band=.05,near_mm=3.):
        p=torch.as_tensor(points,dtype=torch.float64,device='cuda');regions=[];valids=[];nears=[]
        for chunk in p.split(4096):
            inside=[];dist=torch.full((len(chunk),),float('inf'),dtype=torch.float64,device='cuda')
            x=chunk[:,0,None];y=chunk[:,1,None]
            for poly in self.polygons:
                a=poly;b=poly.roll(-1,0);ax=a[:,0];ay=a[:,1];bx=b[:,0];by=b[:,1]
                dy=by-ay;cross=((ay>y)!=(by>y))&(x<((bx-ax)*(y-ay)/torch.where(dy!=0,dy,torch.ones_like(dy))+ax))
                inside.append(cross.sum(1)%2==1)
                v=chunk[:,None,:]-a;d=b-a;t=((v*d).sum(-1)/d.square().sum(-1)).clamp(0,1)
                dist=torch.minimum(dist,(v-t[...,None]*d).square().sum(-1).sqrt().amin(1))
            regions.append(torch.where(inside[1],1,torch.where(inside[0],0,2)).to(torch.int8));valids.append(dist>band);nears.append(dist<=near_mm)
        return tuple(torch.cat(x).cpu().numpy() for x in (regions,valids,nears))
