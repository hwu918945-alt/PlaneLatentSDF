"""Geometry utilities; polygon_sdf/stack_basis adapted from supplied baseline."""
import numpy as np


def stack_basis(axis):
    n = np.asarray(axis, np.float64).copy()
    n /= np.linalg.norm(n)
    anchor = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(anchor, n); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return np.stack([u, v, n], axis=1)


def polygon_sdf(points, polygon):
    p, a = np.asarray(points, np.float64), np.asarray(polygon, np.float64)
    a = a[np.linalg.norm(a-np.roll(a, 1, axis=0), axis=-1) > 1e-10]
    if len(a) < 3:
        raise ValueError('Degenerate polygon')
    b = np.roll(a, -1, axis=0); v = b-a
    length2 = np.sum(v*v, -1)
    t = np.clip(np.sum((p[:, None]-a)*v, -1)/length2, 0, 1)
    d = np.sqrt(np.min(np.sum((p[:, None]-(a+t[..., None]*v))**2, -1), axis=1))
    py = p[:, 1:2]
    crossing = (a[None, :, 1] > py) != (b[None, :, 1] > py)
    den = np.where(np.abs(v[:, 1]) < 1e-14, 1, v[:, 1])
    ix = a[None, :, 0] + (py-a[None, :, 1])*v[None, :, 0]/den
    inside = np.sum(crossing & (p[:, 0:1] < ix), 1) % 2 == 1
    return np.where(inside, -d, d).astype(np.float32)


def contour_labels(q, epi, endo, basis):
    # Compute in bounded chunks: avoid [all grid points,512 contour segments,2].
    de, di = [], []
    for chunk in np.array_split(q, max(1, int(np.ceil(len(q)/1024)))):
        if len(chunk):
            de.append(polygon_sdf(chunk @ basis[:, :2], epi @ basis[:, :2]))
            di.append(polygon_sdf(chunk @ basis[:, :2], endo @ basis[:, :2]))
    de, di = np.concatenate(de), np.concatenate(di)
    region = np.where(di < 0, 1, np.where(de < 0, 0, 2)).astype(np.int64)
    if np.any((di < -1e-5) & (de > 1e-5)):
        raise ValueError('Endo is outside Epi')
    return region, de, di


def contour_report(image, level=0):
    from skimage.measure import find_contours, label
    curves = find_contours(image, level)
    closed = [bool(np.linalg.norm(c[0]-c[-1]) < 1e-4) for c in curves]
    inside = image > level
    border = np.r_[inside[0], inside[-1], inside[:, 0], inside[:, -1]]
    return dict(curves=len(curves), closed_curves=sum(closed), open_curves=len(curves)-sum(closed),
                positive_components=int(label(inside, connectivity=1).max()),
                positive_touches_border=bool(border.any()))
