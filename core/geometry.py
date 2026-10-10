"""Validate continuous closed polygons; never repair missing anatomy silently."""
import numpy as np
from shapely.geometry import Polygon
from .data import polygon_sdf


def clean_polygon(uv):
    uv = np.asarray(uv, dtype=np.float64)
    if uv.ndim != 2 or uv.shape[1] != 2 or not np.isfinite(uv).all():
        raise ValueError('Invalid contour coordinates')
    uv = uv[np.linalg.norm(uv - np.roll(uv, 1, axis=0), axis=1) > 1e-7]
    shape = Polygon(uv) if len(uv) >= 3 else None
    if shape is None or not shape.is_valid or shape.area < 1e-4:
        raise ValueError('Degenerate or self-intersecting polygon')
    return uv


def validate_shell(epi, endo):
    epi, endo = clean_polygon(epi), clean_polygon(endo)
    outer, inner = Polygon(epi), Polygon(endo)
    if not outer.contains(inner) or outer.boundary.distance(inner.boundary) <= 1e-6:
        raise ValueError('Epi/endo must be strictly nested, disjoint closed boundaries')
    return epi, endo


def shell_sdf(points, epi, endo):
    return np.maximum(polygon_sdf(points, epi), -polygon_sdf(points, endo))
