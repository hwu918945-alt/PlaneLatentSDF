# Copied geometry metrics from v11_5static; see provenance/source_files.json.
from __future__ import annotations

from collections import defaultdict
from typing import Any

import numpy as np
from scipy.spatial import cKDTree
from skimage.draw import polygon



PUBLIC_RASTER_SPACING_MM = 0.5


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return np.asarray(value)


def _polygon_mask(
    points: np.ndarray,
    minimum: np.ndarray,
    spacing_mm: float,
    shape: tuple[int, int],
) -> np.ndarray:
    pixels = (points - minimum[None]) / float(spacing_mm)
    rows, columns = polygon(pixels[:, 1], pixels[:, 0], shape=shape)
    mask = np.zeros(shape, dtype=bool)
    mask[rows, columns] = True
    return mask


def _contour_dice(
    observed: np.ndarray,
    components: list[np.ndarray],
    spacing_mm: float,
) -> float:
    extent = np.vstack([observed, *components])
    margin = 2.0 * float(spacing_mm)
    minimum = extent.min(axis=0) - margin
    maximum = extent.max(axis=0) + margin
    shape_xy = np.ceil((maximum - minimum) / float(spacing_mm)).astype(np.int64) + 1
    shape = (int(shape_xy[1]), int(shape_xy[0]))
    observed_mask = _polygon_mask(observed, minimum, spacing_mm, shape)
    predicted_mask = np.zeros(shape, dtype=bool)
    for points in components:
        if len(points) >= 3:
            predicted_mask |= _polygon_mask(points, minimum, spacing_mm, shape)
    denominator = int(observed_mask.sum()) + int(predicted_mask.sum())
    return (
        float(2.0 * np.logical_and(observed_mask, predicted_mask).sum() / denominator)
        if denominator
        else float("nan")
    )


def _segment_components(
    segments: np.ndarray, tolerance_mm: float = 1e-3
) -> list[tuple[np.ndarray, bool]]:
    if len(segments) == 0:
        return []
    node_sum: list[np.ndarray] = []
    node_count: list[int] = []
    key_to_node: dict[tuple[int, int], int] = {}
    segment_nodes: list[tuple[int, int]] = []
    scale = 1.0 / max(float(tolerance_mm), 1e-12)
    for segment in np.asarray(segments, dtype=np.float64):
        ids: list[int] = []
        for point in segment:
            key = tuple(np.rint(point * scale).astype(np.int64).tolist())
            node = key_to_node.get(key)
            if node is None:
                node = len(node_sum)
                key_to_node[key] = node
                node_sum.append(point.copy())
                node_count.append(1)
            else:
                node_sum[node] += point
                node_count[node] += 1
            ids.append(node)
        if ids[0] != ids[1]:
            segment_nodes.append((ids[0], ids[1]))
    nodes = np.asarray(
        [value / float(count) for value, count in zip(node_sum, node_count)],
        dtype=np.float64,
    )
    adjacency: dict[int, set[int]] = defaultdict(set)
    for left, right in set(tuple(sorted(edge)) for edge in segment_nodes):
        adjacency[left].add(right)
        adjacency[right].add(left)
    remaining = set(adjacency)
    components: list[tuple[np.ndarray, bool]] = []
    while remaining:
        seed = min(remaining)
        stack = [seed]
        component_nodes: set[int] = set()
        while stack:
            node = stack.pop()
            if node in component_nodes:
                continue
            component_nodes.add(node)
            stack.extend(adjacency[node] - component_nodes)
        remaining -= component_nodes
        if len(component_nodes) < 3:
            continue
        degrees = {
            node: len(adjacency[node] & component_nodes) for node in component_nodes
        }
        endpoints = sorted(node for node, degree in degrees.items() if degree == 1)
        simple_loop = all(degree == 2 for degree in degrees.values())
        simple_path = len(endpoints) == 2 and all(
            degree in {1, 2} for degree in degrees.values()
        )
        if simple_loop or simple_path:
            start = endpoints[0] if simple_path else min(component_nodes)
            ordered = [start]
            previous, current = -1, start
            while True:
                candidates = sorted((adjacency[current] & component_nodes) - {previous})
                if not candidates:
                    break
                following = candidates[0]
                if (simple_loop and following == start) or following in ordered:
                    break
                ordered.append(following)
                previous, current = current, following
            component = nodes[np.asarray(ordered, dtype=np.int64)]
        else:
            component = nodes[np.asarray(sorted(component_nodes), dtype=np.int64)]
            centre = component.mean(axis=0)
            angle = np.arctan2(component[:, 1] - centre[1], component[:, 0] - centre[0])
            component = component[np.argsort(angle)]
        components.append((component, simple_loop))
    return components


def _mesh_plane_segments(
    vertices: np.ndarray,
    faces: np.ndarray,
    origin: np.ndarray,
    normal: np.ndarray,
) -> np.ndarray:
    triangles = np.asarray(vertices, dtype=np.float64)[np.asarray(faces, dtype=np.int64)]
    distance = np.einsum("fvi,i->fv", triangles - origin[None, None], normal)
    points: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for first, second in ((0, 1), (1, 2), (2, 0)):
        left, right = distance[:, first], distance[:, second]
        difference = left - right
        crossing = (
            (((left <= 0) & (right >= 0)) | ((left >= 0) & (right <= 0)))
            & (np.abs(difference) > 1e-9)
        )
        safe = np.where(crossing, difference, 1.0)
        alpha = np.clip(left / safe, 0.0, 1.0)
        points.append(
            triangles[:, first]
            + alpha[:, None] * (triangles[:, second] - triangles[:, first])
        )
        masks.append(crossing)
    candidates = np.stack(points, axis=1)
    valid = np.stack(masks, axis=1)
    keep = valid.sum(axis=1) >= 2
    if not np.any(keep):
        return np.zeros((0, 2, 3), dtype=np.float64)
    order = np.argsort(~valid[keep], axis=1)
    selected = np.take_along_axis(candidates[keep], order[:, :2, None], axis=1)
    nonzero = np.linalg.norm(selected[:, 1] - selected[:, 0], axis=1) > 1e-7
    return selected[nonzero]


def _slice_metrics(
    observed_3d: np.ndarray,
    segments_3d: np.ndarray,
    origin: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    spacing_mm: float,
    observed_closed: bool,
) -> dict[str, float | None] | None:
    if len(observed_3d) < 3 or len(segments_3d) == 0:
        return None
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    u /= max(np.linalg.norm(u), 1e-12)
    v /= max(np.linalg.norm(v), 1e-12)
    observed = np.stack(
        [(observed_3d - origin) @ u, (observed_3d - origin) @ v], axis=-1
    )
    flat = segments_3d.reshape(-1, 3) - origin
    segment_2d = np.stack([flat @ u, flat @ v], axis=-1).reshape(-1, 2, 2)
    component_rows = _segment_components(segment_2d)
    if not component_rows:
        return None
    components = [points for points, _ in component_rows]
    observed_tree = cKDTree(observed)
    section_tree = cKDTree(np.vstack(components))
    observed_to_section = section_tree.query(observed, k=1)[0]
    means, directed_distances, lengths = [], [], []
    for component, component_closed in component_rows:
        nearest = observed_tree.query(component, k=1)[0]
        means.append(float(nearest.mean()))
        directed_distances.append(np.asarray(nearest, dtype=np.float64))
        length = float(np.linalg.norm(np.diff(component, axis=0), axis=1).sum())
        if component_closed:
            length += float(np.linalg.norm(component[-1] - component[0]))
        lengths.append(max(length, 1e-8))
    weights = np.asarray(lengths, dtype=np.float64)
    section_mean = float(np.sum(np.asarray(means) * weights) / weights.sum())
    symmetric_distances = np.concatenate(
        [np.asarray(observed_to_section, dtype=np.float64), *directed_distances]
    )
    return {
        "mcd_mm": float(0.5 * (observed_to_section.mean() + section_mean)),
        # Symmetric point-sampled HD95. This intentionally replaces the old
        # hd2d_mm field, which was the maximum Hausdorff distance rather than
        # the requested robust 95th-percentile Hausdorff distance.
        "hd95_mm": float(np.percentile(symmetric_distances, 95.0)),
        # Polygon Dice is defined for closed SAX rings. Closing an Open-LAX
        # target would recreate the artificial basal chord removed upstream.
        "contour_dice": (
            _contour_dice(observed, components, spacing_mm)
            if observed_closed
            else None
        ),
    }


def _mean_metrics(rows: list[dict[str, float | None]]) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in ("mcd_mm", "hd95_mm", "contour_dice"):
        values = [row[key] for row in rows if row.get(key) is not None]
        result[key] = float(np.mean(values)) if values else None
    return result


