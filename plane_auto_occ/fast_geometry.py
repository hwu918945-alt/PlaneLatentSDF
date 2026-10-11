"""Float64 compiled polygon classification, with the original geometry contract.

Same stored float32 coordinates, nested Epi/Endo, exact 0.05mm exclusion and
3mm reporting band. No raster labels, distance approximations or fastmath.
"""
import math
import numpy as np
from numba import njit
from .area_pool import PlaneClassifier


@njit(nogil=True,cache=True)
def indexed_polygon(points,edges,ylo,yhi,ray_offsets,ray_edges,xlo,xhi,near_ylo,near_yhi,nx,ny,distance_offsets,distance_edges,reach):
    inside=np.zeros(len(points),np.bool_);distances=np.full(len(points),np.inf)
    for k in range(len(points)):
        x,y=points[k];cross=False
        if ylo<=y<=yhi:
            b=min(len(ray_offsets)-2,int((y-ylo)/(yhi-ylo)*(len(ray_offsets)-1)))
            for z in range(ray_offsets[b],ray_offsets[b+1]):
                ax,ay,bx,by=edges[ray_edges[z]]
                if (ay>y)!=(by>y) and x<(bx-ax)*(y-ay)/(by-ay)+ax:cross=not cross
        inside[k]=cross
        if xlo<=x<=xhi and near_ylo<=y<=near_yhi:
            ix=min(nx-1,int((x-xlo)/(xhi-xlo)*nx));iy=min(ny-1,int((y-near_ylo)/(near_yhi-near_ylo)*ny));b=iy*nx+ix
            minimum=np.inf
            for z in range(distance_offsets[b],distance_offsets[b+1]):
                ax,ay,bx,by=edges[distance_edges[z]]
                if x<min(ax,bx)-reach or x>max(ax,bx)+reach or y<min(ay,by)-reach or y>max(ay,by)+reach:continue
                dx,dy=bx-ax,by-ay;vx,vy=x-ax,y-ay
                t=max(0.,min(1.,(vx*dx+vy*dy)/(dx*dx+dy*dy)))
                minimum=min(minimum,math.sqrt((vx-t*dx)**2+(vy-t*dy)**2))
            distances[k]=minimum
    return inside,distances


def _csr(buckets):
    sizes=np.array([len(x) for x in buckets],np.int64)
    return np.r_[0,np.cumsum(sizes)],np.array([v for bucket in buckets for v in bucket],np.int64)


def _index_polygon(poly,reach):
    edges=np.c_[poly,np.roll(poly,-1,axis=0)];ylo=poly[:,1].min();yhi=poly[:,1].max();ray=[[] for _ in range(128)]
    lo=poly.min(0)-reach;hi=poly.max(0)+reach;nx=max(1,int(np.ceil((hi[0]-lo[0])/reach)));ny=max(1,int(np.ceil((hi[1]-lo[1])/reach)))
    buckets=[[] for _ in range(nx*ny)]
    for j,(ax,ay,bx,by) in enumerate(edges):
        first=max(0,int((min(ay,by)-ylo)/(yhi-ylo)*128));last=min(127,int((max(ay,by)-ylo)/(yhi-ylo)*128))
        for b in range(first,last+1):ray[b].append(j)
        x0=max(0,int((min(ax,bx)-reach-lo[0])/(hi[0]-lo[0])*nx));x1=min(nx-1,int((max(ax,bx)+reach-lo[0])/(hi[0]-lo[0])*nx))
        y0=max(0,int((min(ay,by)-reach-lo[1])/(hi[1]-lo[1])*ny));y1=min(ny-1,int((max(ay,by)+reach-lo[1])/(hi[1]-lo[1])*ny))
        for iy in range(y0,y1+1):
            for ix in range(x0,x1+1):buckets[iy*nx+ix].append(j)
    ro,re=_csr(ray);do,de=_csr(buckets)
    return edges,ylo,yhi,ro,re,lo[0],hi[0],lo[1],hi[1],nx,ny,do,de,reach


# Conservative indexing only eliminates impossible edges; every candidate uses
# the original float64 crossing and exact point-to-segment distance equations.
class CompiledPlaneClassifier(PlaneClassifier):
    def __init__(self,epi,endo):
        super().__init__(epi,endo);self.polygons=tuple(np.ascontiguousarray(edge[0],np.float64) for edge in self.edges);self.indices={}
    def classify(self,points,band=.05,near_mm=3.):
        reach=max(float(band),float(near_mm))
        if reach<=0:reach=1.
        if reach not in self.indices:self.indices[reach]=tuple(_index_polygon(p,reach) for p in self.polygons)
        results=[indexed_polygon(np.ascontiguousarray(points,np.float64),*index) for index in self.indices[reach]]
        distance=np.minimum(results[0][1],results[1][1])
        return np.where(results[1][0],1,np.where(results[0][0],0,2)).astype(np.int8),distance>band,distance<=near_mm
