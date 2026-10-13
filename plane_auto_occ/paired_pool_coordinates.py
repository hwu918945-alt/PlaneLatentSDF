"""Validate sampling units in their final float32 training coordinates.

The supplied physical-mm geometry core remains unchanged. Uniform proposals
are rejected before publication; boundary retries preserve the original anchor,
scale assignment and two-sided sampling unit. Every retry is recorded.
"""
import numpy as np


VERSION = 'float32_world_valid_unique_v2'


def map_world(uv, scale, basis, position):
    return (uv / scale @ basis[:, :2].T + float(position) * basis[:, 2]).astype(np.float32)


def repair_coordinates(plane, q, source, banks, scale, basis, position, rng):
    """Return valid unique coordinates, final labels and an explicit retry audit."""
    region, valid, _ = plane.classify(q.astype(np.float64) @ basis[:, :2] * scale)
    expected = np.array(getattr(plane, 'source_regions', [0, 2, 0, 1]), np.int8)
    boundary = source < 4
    invalid = ~valid
    invalid[boundary] |= region[boundary] != expected[source[boundary]]
    _, first = np.unique(q, axis=0, return_index=True)
    duplicate = np.ones(len(q), bool)
    duplicate[first] = False
    audit = dict(version=VERSION, initial_invalid_by_source={
        str(k): int((invalid & (source == k)).sum()) for k in range(5)},
        initial_duplicate_by_source={str(k): int((duplicate & (source == k)).sum()) for k in range(5)},
        boundary_pairs_retried=[], roi_points_retried=0,
        roi_rejected_after_mapping=0, roi_rejected_duplicates=0)
    if not (invalid | duplicate).any():
        return q, region, audit

    # Keep complete unaffected pairs and valid ROI points. Reserve their final
    # coordinates so later retries cannot collide with another sampling unit.
    bad = invalid | duplicate
    offset = 0
    pairs_to_retry = []
    for b, bank in enumerate(banks):
        count = len(bank['q'])
        units = np.flatnonzero(bad[offset:offset + 2 * count].reshape(-1, 2).any(1))
        for unit in units:
            rows = np.arange(offset + 2 * unit, offset + 2 * unit + 2)
            bad[rows] = True
            pairs_to_retry.append((b, int(unit), rows))
        offset += 2 * count
    used = {tuple(row) for row in q[~bad]}

    for b, unit, rows in pairs_to_retry:
        bank = banks[b]
        curve, edge, length, _, cumulative = plane.edges[b]
        arc = bank['arc'][unit]
        segment = min(np.searchsorted(cumulative, arc, side='right') - 1, len(curve) - 1)
        anchor = curve[segment] + (arc - cumulative[segment]) / length[segment] * edge[segment]
        direction = bank['q'][unit, 0].astype(np.float64) - bank['q'][unit, 1]
        direction /= np.linalg.norm(direction)
        initial_delta = float(bank['delta'][unit])
        found = False
        for angle in (0, 15, -15, 30, -30, 45, -45):
            radians = np.deg2rad(angle)
            rotation = np.array([[np.cos(radians), -np.sin(radians)],
                                 [np.sin(radians), np.cos(radians)]])
            normal = direction @ rotation.T
            for factor in (1., .99999, .9999, .999, .99, .98, .95, .9, .7, .4, .2):
                delta = min(initial_delta * factor, .4 * bank['clearance'][unit])
                uv = np.stack([anchor + delta * normal, anchor - delta * normal]).astype(np.float32)
                candidate = map_world(uv, scale, basis, position)
                labels, accepted, _ = plane.classify(candidate.astype(np.float64) @ basis[:, :2] * scale)
                keys = [tuple(row) for row in candidate]
                if (accepted.all() and np.array_equal(labels, expected[2*b:2*b+2])
                        and keys[0] != keys[1] and all(key not in used for key in keys)):
                    q[rows] = candidate
                    region[rows] = labels
                    used.update(keys)
                    bank['q'][unit] = uv
                    bank['delta'][unit] = delta
                    bank['capped'][unit] = delta < bank['requested'][unit] - 1e-8
                    bank['shrunk'][unit] |= delta < initial_delta
                    bank['rotated'][unit] |= angle != 0
                    audit['boundary_pairs_retried'].append(dict(
                        boundary=b, unit=unit, arc_mm=float(arc),
                        old_delta_mm=initial_delta, new_delta_mm=float(delta), angle_degrees=angle,
                        scale_id=int(bank['scale_id'][unit]), anchor_preserved=True))
                    found = True
                    break
            if found:
                break
        if not found:
            raise ValueError(f'Unsupported final float32 pair at boundary {b}, anchor {arc}; no anchor replacement')

    remaining = np.flatnonzero(bad & ~boundary)
    audit['roi_points_retried'] = len(remaining)
    for _ in range(64):
        if not len(remaining):
            break
        candidate = map_world(plane.uniform(len(remaining), rng), scale, basis, position)
        labels, accepted, _ = plane.classify(candidate.astype(np.float64) @ basis[:, :2] * scale)
        audit['roi_rejected_after_mapping'] += int((~accepted).sum())
        filled = 0
        for index in np.flatnonzero(accepted):
            key = tuple(candidate[index])
            if key in used:
                audit['roi_rejected_duplicates'] += 1
                continue
            row = remaining[filled]
            q[row] = candidate[index]
            region[row] = labels[index]
            used.add(key)
            filled += 1
        remaining = remaining[filled:]
    if len(remaining):
        raise ValueError('Insufficient unique valid ROI proposals after float32 mapping')
    return q, region, audit
