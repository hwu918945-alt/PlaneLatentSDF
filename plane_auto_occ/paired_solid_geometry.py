"""Real multi-ring solid boundaries: paired offsets without invented shell rings."""
import numpy as np
import shapely
from shapely.geometry import Polygon
from .solid import SolidClassifier
from .paired_geometry import Plane


class SolidPlane:
    source_regions = [0, 1, 0, 1]
    anchors = Plane.anchors
    uniform = Plane.uniform

    def __init__(self, rings, lower, upper, band=.05):
        classifier=SolidClassifier(rings)
        self.shape=classifier.shape
        self.rings=classifier.rings
        self.lower=np.asarray(lower,dtype=float);self.upper=np.asarray(upper,dtype=float);self.band=float(band)
        curves=[];vectors=[];lengths=[];normals=[]
        polygons=[Polygon(r) for r in self.rings]
        for i,curve in enumerate(self.rings):
            vector=np.roll(curve,-1,axis=0)-curve;length=np.linalg.norm(vector,axis=1)
            area=(curve[:,0]*np.roll(curve[:,1],-1)-np.roll(curve[:,0],-1)*curve[:,1]).sum()/2
            normal=np.column_stack([vector[:,1],-vector[:,0]])/length[:,None]*np.sign(area)
            depth=sum(shapely.contains_xy(p,curve[0,0],curve[0,1]) for j,p in enumerate(polygons) if j!=i)
            if depth%2:normal=-normal
            curves.append(curve);vectors.append(vector);lengths.append(length);normals.append(normal)
        c=np.concatenate(curves);v=np.concatenate(vectors);length=np.concatenate(lengths);normal=np.concatenate(normals)
        # Each vector ends on its own ring; concatenation never joins rings.
        edge=(c,v,length,normal,np.r_[0,np.cumsum(length)])
        self.edges=[edge,edge];self.curves=[c,c]

    def classify(self,q):
        q=np.asarray(q,dtype=np.float64)
        inside=shapely.contains_xy(self.shape,q[:,0],q[:,1])
        distance=shapely.distance(shapely.points(q),self.shape.boundary)
        valid=(distance>self.band)&(q>=self.lower).all(1)&(q<=self.upper).all(1)
        return np.where(inside,0,1).astype(np.int8),valid,np.column_stack([distance,distance])

    def clearance(self,anchors,normals,requested=None):
        """Nearest other real boundary along the solid-interior normal ray."""
        reach=float(np.linalg.norm(self.upper-self.lower)*2+1)
        result=np.full(len(anchors),np.inf)
        for start in range(0,len(anchors),4096):
            a=anchors[start:start+4096];n=normals[start:start+4096]
            lines=shapely.linestrings(np.stack([a-reach*n,a+reach*n],axis=1))
            intersections=shapely.intersection(lines,self.shape.boundary)
            xy,owner=shapely.get_coordinates(intersections,return_index=True)
            signed=((xy-a[owner])*n[owner]).sum(1)
            distance=-signed
            accepted=signed < -1e-6
            np.minimum.at(result[start:start+len(a)],owner[accepted],distance[accepted])
        # Missing intersections describe unsupported individual rays. Let the
        # existing same-anchor retries/audited rejection handle those units.
        return result

    def pairs(self,boundary,count,rng,scales=(.25,.75,2.),weights=(.5,.3,.2),adaptive=True,
              _scale_id=None,_requested=None,_round=0):
        if boundary not in (0,1) or count<=0:raise ValueError('Invalid independent boundary bank')
        if len(scales)!=len(weights) or min(scales)<=0 or min(weights)<0 or not np.isclose(sum(weights),1):
            raise ValueError('Invalid scales or quotas')
        a,n,arc=self.anchors(boundary,count,rng)
        quota=np.floor(np.asarray(weights)*count).astype(int);quota[np.argmax(weights)]+=count-quota.sum()
        scale_id=np.repeat(np.arange(len(scales)),quota);rng.shuffle(scale_id)
        requested=np.asarray(scales)[scale_id]*rng.uniform(.8,1.2,count)
        if _scale_id is not None:scale_id=_scale_id.copy();requested=_requested.copy()
        h=self.clearance(a,n,requested);ray_valid=np.isfinite(h)&(h>0)
        missing_rays=int((~ray_valid).sum())
        delta=np.where(ray_valid,np.minimum(requested,.4*h) if adaptive else requested,0.)
        def candidate(ids):
            return np.stack([a[ids]-delta[ids,None]*n[ids],a[ids]+delta[ids,None]*n[ids]],axis=1).astype(np.float32)
        q=candidate(np.arange(count));r,ok,_=self.classify(q.reshape(-1,2))
        good=ok.reshape(-1,2).all(1)&(r.reshape(-1,2)==[0,1]).all(1)
        initial_good=good.copy();shrunk=np.zeros(count,bool);rotated=np.zeros(count,bool)
        rejected=[]
        if adaptive:
            for _ in range(4):
                ids=np.flatnonzero(~good)
                if not len(ids):break
                delta[ids]*=.7;shrunk[ids]=True;q[ids]=candidate(ids)
                r,ok,_=self.classify(q[ids].reshape(-1,2))
                good[ids]=ok.reshape(-1,2).all(1)&(r.reshape(-1,2)==[0,1]).all(1)
            for angle in (0,15,-15,30,-30,45,-45):
                ids=np.flatnonzero(~good)
                if not len(ids):break
                rad=np.deg2rad(angle);rot=np.array([[np.cos(rad),-np.sin(rad)],[np.sin(rad),np.cos(rad)]])
                normal=n[ids]@rot.T
                rotated_h=self.clearance(a[ids],normal,requested[ids])
                ray_valid=np.isfinite(rotated_h)&(rotated_h>0)
                missing_rays+=int((~ray_valid).sum())
                for factor in (1.,.7,.4,.2):
                    dd=np.where(ray_valid,np.minimum(requested[ids]*factor,.4*rotated_h),0.)
                    pp=np.stack([a[ids]-dd[:,None]*normal,a[ids]+dd[:,None]*normal],axis=1).astype(np.float32)
                    rr,valid,_=self.classify(pp.reshape(-1,2))
                    success=ray_valid&valid.reshape(-1,2).all(1)&(rr.reshape(-1,2)==[0,1]).all(1)&~good[ids]
                    chosen=ids[success];q[chosen]=pp[success];delta[chosen]=dd[success]
                    h[chosen]=rotated_h[success]
                    good[chosen]=True;rotated[chosen]=angle!=0
            if not good.all():
                bad=np.flatnonzero(~good)
                rejected=[dict(arc_mm=float(arc[i]),clearance_mm=float(h[i]) if np.isfinite(h[i]) else None,round=_round) for i in bad]
                if _round>=12:raise ValueError('Insufficient real-solid boundary support after audited anchor rejection')
                # A solid contour can contain sub-band spikes/corners with no
                # admissible bilateral pair. Condition arclength sampling on
                # actual geometric support; preserve each rejected unit's scale
                # and jittered requested offset, and record every rejection.
                replacement=self.pairs(boundary,len(bad),rng,scales,weights,adaptive,
                                       _scale_id=scale_id[bad],_requested=requested[bad],_round=_round+1)
                for array,name in [(q,'q'),(delta,'delta'),(h,'clearance'),(arc,'arc'),
                                   (shrunk,'shrunk'),(rotated,'rotated')]:array[bad]=replacement[name]
                good[bad]=True;rejected.extend(replacement['unsupported_anchors_rejected'])
                missing_rays+=replacement['missing_boundary_ray_attempts']
        return dict(q=q,good=good,initial_good=initial_good,requested=requested,delta=delta,
                    scale_id=scale_id,clearance=h,arc=arc,capped=delta<requested-1e-8,
                    shrunk=shrunk,rotated=rotated,unsupported_anchors_rejected=rejected,
                    missing_boundary_ray_attempts=missing_rays)
