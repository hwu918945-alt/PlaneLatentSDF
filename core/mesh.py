import numpy as np
from skimage.measure import marching_cubes


def stack_basis(axis):
    n = np.asarray(axis, np.float64)
    n /= np.linalg.norm(n)
    anchor = np.eye(3)[np.argmin(np.abs(n))]
    u = np.cross(anchor, n); u /= np.linalg.norm(u)
    v = np.cross(n, u)
    return np.stack([u, v, n], axis=1)  # world q = stack coordinates @ basis.T


def volume_to_mesh(volume, lower, upper, basis, center_mm, scale_mm):
    if not np.isfinite(volume).all():
        raise ValueError('Nonfinite scalar volume')
    if not float(volume.min()) < 0 < float(volume.max()):
        raise ValueError('No sign change: cannot extract level zero')
    spacing = (np.asarray(upper) - np.asarray(lower)) / (np.asarray(volume.shape) - 1)
    # Verified on negative-inside analytic sphere: this winding points outward.
    vertices, faces, _, _ = marching_cubes(volume, level=0, spacing=tuple(spacing), gradient_direction='descent')
    vertices_stack = vertices + np.asarray(lower)
    vertices_q = vertices_stack @ np.asarray(basis).T
    vertices_mm = vertices_q * scale_mm + np.asarray(center_mm)
    edges = np.sort(np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]]), axis=1)
    _, counts = np.unique(edges, axis=0, return_counts=True)
    at_border = np.any((vertices_stack <= np.asarray(lower) + spacing * 0.1) |
                       (vertices_stack >= np.asarray(upper) - spacing * 0.1), axis=1)
    contacts = (vertices_stack <= np.asarray(lower)+spacing*.1)|(vertices_stack >= np.asarray(upper)-spacing*.1)
    return vertices_mm, faces, {'vertices': int(len(vertices)), 'faces': int(len(faces)),
                                'boundary_edges': int(np.sum(counts == 1)),
                                'nonmanifold_edges': int(np.sum(counts > 2)),
                                'watertight_edge_test': bool(np.all(counts == 2)),
                                'vertices_touching_domain': int(at_border.sum()),
                                'vertices_on_lateral_boundary': int(contacts[:,:2].any(1).sum()),
                                'vertices_on_axial_boundary': int(contacts[:,2].sum()),
                                'lateral_domain_truncated': bool(contacts[:,:2].any()),
                                'field_min': float(volume.min()), 'field_max': float(volume.max())}


def write_obj(path, vertices, faces):
    with open(path, 'w') as f:
        f.write('# vertices are in aligned world millimetres\n')
        for x, y, z in vertices:
            f.write(f'v {x:.8f} {y:.8f} {z:.8f}\n')
        for a, b, c in faces + 1:
            f.write(f'f {a} {b} {c}\n')
