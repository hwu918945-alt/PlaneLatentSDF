"""Reference sampling in a labelled 2-D observed plane, coordinates in mm.

This is a geometry experiment/reference implementation, not a repository patch.
Dependencies: numpy, scipy, shapely>=2. Empty/non-nested shells raise explicitly.
Each pair is one sampling unit. Its two coordinates are validated independently.
"""
import numpy as np
import shapely
from shapely.geometry import Polygon
from scipy.spatial import cKDTree


class Plane:
    def __init__(self, epi, endo, lower, upper, band=.05):
        self.curves=[]
        for curve in (epi, endo):
            curve=np.asarray(curve, dtype=np.float64)
            curve=curve[np.linalg.norm(curve-np.roll(curve,1,axis=0),axis=1)>1e-9]
            shape=Polygon(curve)
            if not shape.is_valid or shape.area<=0: raise ValueError('Invalid contour')
            self.curves.append(curve)
        self.polys=[Polygon(c) for c in self.curves]
        if not self.polys[0].contains(self.polys[1]): raise ValueError('Non-nested contours')
        self.lower=np.asarray(lower); self.upper=np.asarray(upper); self.band=band
        self.edges=[]
        for c in self.curves:
            v=np.roll(c,-1,axis=0)-c; length=np.linalg.norm(v,axis=1)
            # CCW curve: right-hand normal points outside the enclosed polygon.
            area=(c[:,0]*np.roll(c[:,1],-1)-np.roll(c[:,0],-1)*c[:,1]).sum()/2
            normal=np.column_stack([v[:,1],-v[:,0]])/length[:,None]*np.sign(area)
            self.edges.append((c,v,length,normal,np.r_[0,np.cumsum(length)]))

    def classify(self, q):
        q=np.asarray(q,dtype=np.float64); points=shapely.points(q)
        inside=[shapely.contains_xy(p,q[:,0],q[:,1]) for p in self.polys]
        d=np.column_stack([shapely.distance(points,p.boundary) for p in self.polys])
        r=np.where(inside[1],1,np.where(inside[0],0,2)).astype(np.int8)
        valid=(d.min(1)>self.band)&(q>=self.lower).all(1)&(q<=self.upper).all(1)
        return r,valid,d

    def anchors(self, boundary, count, rng=None):
        c,v,length,n,cum=self.edges[boundary]
        # Stratified equal-arclength anchors, randomized within each stratum.
        jitter=np.full(count,.5) if rng is None else rng.random(count)
        s=(np.arange(count)+jitter)/count*cum[-1]
        ids=np.minimum(np.searchsorted(cum,s,side='right')-1,len(c)-1)
        a=c[ids]+((s-cum[ids])/length[ids])[:,None]*v[ids]
        return a,n[ids],s

    def uniform(self, count, rng, region=None):
        lower,upper=self.lower,self.upper
        if region is not None and region<2:
            c=self.curves[region];lower=c.min(0);upper=c.max(0)
        parts=[]; remaining=count; proposed=0
        while remaining:
            size=max(4096,remaining*2);proposed+=size
            if proposed>20000000: raise ValueError('Insufficient region area')
            q=rng.uniform(lower,upper,(size,2)).astype(np.float32)
            r,valid,_=self.classify(q)
            if region is not None: valid&=r==region
            q=q[valid][:remaining];parts.append(q);remaining-=len(q)
        return np.concatenate(parts)

    def pairs(self, boundary, count, rng, scales=(.25,.75,2.), weights=(.5,.3,.2), adaptive=True):
        """Return count pairs [myocardium, negative], with a shared offset.

        The nearest OTHER contour distance is a conservative local clearance,
        not a measured normal-direction anatomical wall thickness. Cap at 0.4h.
        Scale quotas are preserved and failures retried at the SAME anchor first.
        Every final float32 coordinate is relabelled; no normal-derived labels.
        """
        if boundary not in (0,1) or count<=0: raise ValueError('Invalid pair request')
        if len(scales)!=len(weights) or min(scales)<=0 or min(weights)<0 or not np.isclose(sum(weights),1):
            raise ValueError('Positive scales and normalized matching weights required')
        a,n,s=self.anchors(boundary,count,rng)
        quota=np.floor(np.asarray(weights)*count).astype(int)
        quota[np.argmax(weights)]+=count-quota.sum()
        scale_id=np.repeat(np.arange(len(scales)),quota);rng.shuffle(scale_id)
        # Continuous jitter prevents duplicate coordinates after capping.
        requested=np.asarray(scales)[scale_id]*rng.uniform(.8,1.2,count)
        h=shapely.distance(shapely.points(a),self.polys[1-boundary].boundary)
        delta=np.minimum(requested,.4*h) if adaptive else requested.copy()
        sign=-1 if boundary==0 else 1
        def candidate(ids):
            positive=a[ids]+sign*delta[ids,None]*n[ids]
            negative=a[ids]-sign*delta[ids,None]*n[ids]
            return np.stack([positive,negative],axis=1).astype(np.float32)
        q=candidate(np.arange(count));r,ok,_=self.classify(q.reshape(-1,2))
        expected=np.array([0,2 if boundary==0 else 1])
        good=ok.reshape(-1,2).all(1)&(r.reshape(-1,2)==expected).all(1)
        initial_good=good.copy();shrunk=np.zeros(count,bool)
        rotated=np.zeros(count,bool)
        if adaptive:
            # Concave corners and discretized normals can invalidate same-boundary clearance.
            for _ in range(4):
                ids=np.flatnonzero(~good)
                if not len(ids):break
                delta[ids]*=.7;shrunk[ids]=True
                q[ids]=candidate(ids);r,ok,_=self.classify(q[ids].reshape(-1,2))
                good[ids]=ok.reshape(-1,2).all(1)&(r.reshape(-1,2)==expected).all(1)
            # At a corner, reducing distance can put both samples in the excluded
            # same-boundary band. Search small direction changes at the SAME anchor.
            # Restore a useful candidate distance, then verify exact geometry again.
            for angle in (0,15,-15,30,-30,45,-45):
                ids=np.flatnonzero(~good)
                if not len(ids):break
                rad=np.deg2rad(angle);rot=np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
                normal=n[ids]@rot.T
                for factor in (1.,.7,.4,.2):
                    dd=np.minimum(requested[ids]*factor,.4*h[ids])
                    pp=np.stack([a[ids]+sign*dd[:,None]*normal,a[ids]-sign*dd[:,None]*normal],axis=1).astype(np.float32)
                    rr,valid,_=self.classify(pp.reshape(-1,2))
                    success=valid.reshape(-1,2).all(1)&(rr.reshape(-1,2)==expected).all(1)&~good[ids]
                    chosen=ids[success];q[chosen]=pp[success];delta[chosen]=dd[success]
                    good[chosen]=True;rotated[chosen]=angle!=0
            if not good.all():
                # Do not replace unsupported locations silently: reject geometry.
                raise ValueError(f'{(~good).sum()} unsupported anchors: inspect contour/band')
        return dict(q=q,good=good,initial_good=initial_good,requested=requested,
                    delta=delta,scale_id=scale_id,clearance=h,arc=s,
                    capped=delta<requested-1e-8,shrunk=shrunk,rotated=rotated)


def pool_draw(pair_epi, pair_endo, uniform, visit, budgets=(3072,3072,20480), seed=19):
    """budgets are PAIR counts for Epi/Endo and POINT count for uniform.

    Default total=2*3072+2*3072+20480=32768. Each bank has 8x capacity.
    Pair arrays: [8*pair_count, 2, 2]; uniform: [8*point_count, 2].
    Regenerate all banks after 80 visits using a new generation seed.
    """
    items=[]
    for bank,n,key in zip((pair_epi,pair_endo,uniform),budgets,range(3)):
        if len(bank)!=8*n: raise ValueError('Expected eight-draw capacity')
        cycle,offset=divmod(visit,8)
        order=np.random.default_rng(np.random.SeedSequence([seed,key,cycle])).permutation(len(bank))
        part=bank[order[offset*n:(offset+1)*n]]
        items.append(part.reshape(-1,2))
    return np.concatenate(items)
