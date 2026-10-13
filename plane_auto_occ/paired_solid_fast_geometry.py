"""Compiled exact segment predicates; GEOS refinement preserves capped coordinates.

No rasterization, SDF interpolation, changed proposals, or changed RNG draws.
The classifier reports infinity for distances beyond its conservative search
radius; the band-valid predicate remains exact. Those far distances are not
used in sampling, labels, clearance, losses, or recorded training statistics.
"""
import numpy as np
import shapely
from numba import njit
from .fast_geometry import _index_polygon,indexed_polygon
from .paired_solid_geometry import SolidPlane


@njit(nogil=True,cache=True)
def ray_hits(anchors,normals,starts,vectors):
    result=np.full(len(anchors),np.inf);ambiguous=np.zeros(len(anchors),np.bool_)
    for k in range(len(anchors)):
        ax,ay=anchors[k];dx,dy=-normals[k]
        for j in range(len(starts)):
            sx,sy=starts[j];ex,ey=vectors[j];vx,vy=sx-ax,sy-ay
            denominator=dx*ey-dy*ex
            if abs(denominator)<1e-13:
                if abs(vx*dy-vy*dx)<1e-10:ambiguous[k]=True
                continue
            t=(vx*ey-vy*ex)/denominator
            u=(vx*dy-vy*dx)/denominator
            if t>1e-6:
                if abs(u)<1e-10 or abs(u-1.)<1e-10:ambiguous[k]=True
                if 0.<=u<=1. and t<result[k]:result[k]=t
            elif abs(t-1e-6)<1e-9 and 0.<=u<=1.:ambiguous[k]=True
    return result,ambiguous


class CompiledSolidPlane(SolidPlane):
    def __init__(self,rings,lower,upper,band=.05):
        super().__init__(rings,lower,upper,band)
        self.indices=tuple(_index_polygon(np.ascontiguousarray(r,np.float64),max(1.,2*self.band)) for r in self.rings)

    def classify(self,q):
        q=np.ascontiguousarray(q,dtype=np.float64)
        inside=np.zeros(len(q),bool);distance=np.full(len(q),np.inf)
        for index in self.indices:
            contained,dist=indexed_polygon(q,*index)
            inside^=contained;distance=np.minimum(distance,dist)
        # Borderline decisions use the original GEOS operations, eliminating
        # last-bit changes to the exclusion predicate and proposal acceptance.
        uncertain=np.isfinite(distance)&(np.abs(distance-self.band)<1e-9)
        if uncertain.any():
            distance[uncertain]=shapely.distance(shapely.points(q[uncertain]),self.shape.boundary)
            inside[uncertain]=shapely.contains_xy(self.shape,q[uncertain,0],q[uncertain,1])
        valid=(distance>self.band)&(q>=self.lower).all(1)&(q<=self.upper).all(1)
        return np.where(inside,0,1).astype(np.int8),valid,np.column_stack([distance,distance])

    def clearance(self,anchors,normals,requested=None):
        c,v,_,_,_=self.edges[0]
        h,ambiguous=ray_hits(np.ascontiguousarray(anchors),np.ascontiguousarray(normals),c,v)
        if requested is None:
            # Public/reference comparison asks for the exact GEOS result.
            refine=np.ones(len(h),bool)
        else:
            # Uncapped offsets are determined entirely by requested. Refine
            # all possibly active caps, degenerate rays, and endpoint ambiguity
            # so final float32 q/delta and every RNG branch equal the reference.
            refine=ambiguous|~np.isfinite(h)|(h<=np.asarray(requested)/.4+1e-8)
        if refine.any():h[refine]=super().clearance(anchors[refine],normals[refine],None)
        return h
