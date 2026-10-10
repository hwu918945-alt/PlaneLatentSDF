from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from scipy import ndimage

from .geometry import ShellTopology


STATIC_CACHE_VERSION = "v11_5static_endpoint_case_v2_myo_training_aux"
STATIC_MANIFEST_VERSION = "v11_5static_endpoint_manifest_v2"
SHARED_TEMPLATE_NORMALIZATION = "endpoint_contour_sphere_v1"
FORBIDDEN_CACHE_FIELDS = {
    "images",
    "masks",
    "myo_views",
    "reliability",
    "slice_valid",
    "frame_indices",
}


@dataclass
class StaticEndpointCase:
    case_id: str
    split: str
    contours: torch.Tensor
    contour_valid: torch.Tensor
    contour_closed: torch.Tensor
    view_ids: torch.Tensor
    slice_positions: torch.Tensor
    plane_origins: torch.Tensor
    plane_normals: torch.Tensor
    plane_u: torch.Tensor
    plane_v: torch.Tensor
    training_myo_views: torch.Tensor
    endpoint_cavity_volume_ml: torch.Tensor
    initial_vertices: torch.Tensor
    faces: torch.Tensor
    topology: ShellTopology
    center: np.ndarray
    scale: float
    canonical_transform: dict[str, Any]
    orientation: np.ndarray
    loss_geometry: dict[str, torch.Tensor] | None = None


def _json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_static_manifest(
    config: dict[str, Any], output: str | Path | None = None
) -> dict[str, Any]:
    """Create a manifest that exposes only ED/ES endpoint provenance."""

    data_cfg = config["data"]
    source = _json(data_cfg["source_manifest"])
    source_root = Path(data_cfg["source_cache_root"])
    records = []
    for row in source["records"]:
        case_id = str(row["case_id"]).zfill(3)
        records.append(
            {
                "case_id": case_id,
                "split": str(row["split"]),
                "source_full_cycle_cache": str(
                    source_root / case_id / "full_cycle_case.npz"
                ),
                "endpoint_cavity_volume_ml": row["endpoint_cavity_volume_ml"],
                "center": row["center"],
                "scale": row["scale"],
                "canonical_transform": row["canonical_transform"],
                "case_orientation": row["case_orientation"],
            }
        )
    counts: dict[str, int] = {}
    for row in records:
        counts[row["split"]] = counts.get(row["split"], 0) + 1
    manifest = {
        "version": STATIC_MANIFEST_VERSION,
        "case_count": len(records),
        "split_counts": counts,
        "phase_order": ["ED", "ES"],
        "online_inputs": "ED/ES epi+endo contours and slice plane geometry only",
        "training_auxiliary": (
            "ED/ES myocardium SDF/mask view for cross-view BYOL only; "
            "not passed at inference"
        ),
        "excluded": [
            "cine images",
            "intermediate frames",
            "alignment reliability weights",
            "pseudo masks",
            "motion targets",
        ],
        "source_manifest_version": source.get("version"),
        "records": records,
    }
    path = Path(output or data_cfg["manifest"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def load_manifest(path: str | Path, split: str | None = None) -> list[dict[str, Any]]:
    payload = _json(path)
    if payload.get("version") != STATIC_MANIFEST_VERSION:
        raise ValueError(
            f"static manifest version={payload.get('version')!r}, "
            f"expected={STATIC_MANIFEST_VERSION!r}"
        )
    return [
        row
        for row in payload["records"]
        if split is None or str(row["split"]) == str(split)
    ]


def static_cache_path(config: dict[str, Any], case_id: str) -> Path:
    return (
        Path(config["data"]["cache_root"])
        / str(case_id).zfill(3)
        / "endpoint_case.npz"
    )


def endpoint_contour_normalization(
    contours: np.ndarray, contour_valid: np.ndarray
) -> tuple[np.ndarray, float]:
    value = np.asarray(contours, dtype=np.float32)
    valid = np.asarray(contour_valid, dtype=bool)
    if value.ndim != 5 or value.shape[:1] != (2,) or value.shape[2] != 2:
        raise ValueError(f"endpoint contours must be [2,K,2,P,3], got {value.shape}")
    rows = [
        value[phase, slice_id, surface]
        for phase in range(2)
        for slice_id in range(value.shape[1])
        for surface in range(2)
        if valid[phase, slice_id, surface]
    ]
    if not rows:
        raise ValueError("no valid ED/ES epi+endo contours")
    points = np.concatenate(rows, axis=0).astype(np.float64)
    center = points.mean(axis=0)
    radius = float(np.linalg.norm(points - center[None], axis=1).max())
    if not np.isfinite(points).all() or not np.isfinite(radius) or radius <= 1e-6:
        raise ValueError("invalid endpoint contour normalization")
    return center.astype(np.float32), radius


def prepare_static_case(
    record: dict[str, Any], config: dict[str, Any], force: bool = False
) -> Path:
    """Strip a v11_5 cache to a normalized ED/ES contour-only cache."""

    case_id = str(record["case_id"]).zfill(3)
    output = static_cache_path(config, case_id)
    if output.exists() and not force:
        with np.load(output, allow_pickle=False) as payload:
            metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
            if metadata.get("version") == STATIC_CACHE_VERSION:
                return output
    source_path = Path(record["source_full_cycle_cache"])
    if not source_path.exists():
        raise FileNotFoundError(f"source v11_5 cache missing for case {case_id}: {source_path}")
    with np.load(source_path, allow_pickle=False) as source:
        es_index = int(np.asarray(source["es_index"]).item())
        phase_indices = np.asarray([0, es_index], dtype=np.int64)
        contours = np.asarray(source["contours"], dtype=np.float32)[phase_indices]
        valid = np.asarray(source["contour_valid"], dtype=bool)[phase_indices]
        origins = np.asarray(source["plane_origins"], dtype=np.float32)[phase_indices]
        normals = np.asarray(source["plane_normals"], dtype=np.float32)[phase_indices]
        plane_u = np.asarray(source["plane_u"], dtype=np.float32)[phase_indices]
        plane_v = np.asarray(source["plane_v"], dtype=np.float32)[phase_indices]
        view_ids = np.asarray(source["view_ids"], dtype=np.int64)
        contour_closed = np.asarray(source["contour_closed"], dtype=bool)
        endpoint_masks = np.asarray(source["masks"], dtype=np.int16)[phase_indices]

    # Training-only myocardium view, copied exactly from the original Base
    # auxiliary construction but restricted to ED/ES.  It is never accepted by
    # the inference path and is not an online method input.
    training_myo_views = np.zeros((*endpoint_masks.shape, 3), dtype=np.float32)
    for index in np.ndindex(endpoint_masks.shape[:2]):
        mask = endpoint_masks[index]
        endo = mask == 1
        epi = (mask == 1) | (mask == 2)
        myo = mask == 2
        for channel, binary in enumerate((epi, endo)):
            outside = ndimage.distance_transform_edt(~binary)
            inside = ndimage.distance_transform_edt(binary)
            training_myo_views[index + (slice(None), slice(None), channel)] = (
                np.clip(outside - inside, -16.0, 16.0) / 16.0
            )
        training_myo_views[index + (slice(None), slice(None), 2)] = myo.astype(
            np.float32
        )
    training_myo_views = np.moveaxis(training_myo_views, -1, 2)

    offset, relative_scale = endpoint_contour_normalization(contours, valid)
    contours = (contours - offset[None, None, None, None]) / relative_scale
    origins = (origins - offset[None, None]) / relative_scale
    plane_u = plane_u / relative_scale
    plane_v = plane_v / relative_scale
    center = np.asarray(record["center"], dtype=np.float32)
    source_scale = float(record["scale"])
    center = center + source_scale * offset
    scale = source_scale * relative_scale
    positions = np.zeros_like(view_ids, dtype=np.float32)
    sax_ids = np.flatnonzero(view_ids == 0)
    positions[sax_ids] = np.linspace(-1.0, 1.0, len(sax_ids), dtype=np.float32)
    metadata = {
        "version": STATIC_CACHE_VERSION,
        "case_id": case_id,
        "split": str(record["split"]),
        "phase_order": ["ED", "ES"],
        "source_es_index_ed_first": es_index,
        "source_cache": str(source_path),
        "normalization": SHARED_TEMPLATE_NORMALIZATION,
        "input_contract": "endpoint epi+endo contours plus plane geometry",
        "reliability_present": False,
        "training_auxiliary": "ED/ES myocardium SDF/mask view; never an inference input",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        contours=contours.astype(np.float16),
        contour_valid=valid,
        contour_closed=contour_closed,
        view_ids=view_ids,
        slice_positions=positions,
        plane_origins=origins.astype(np.float32),
        plane_normals=normals.astype(np.float32),
        plane_u=plane_u.astype(np.float32),
        plane_v=plane_v.astype(np.float32),
        training_myo_views=training_myo_views.astype(np.float16),
        center=center.astype(np.float32),
        scale=np.asarray(scale, dtype=np.float32),
        metadata=np.asarray(json.dumps(metadata)),
    )
    output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    return output


@lru_cache(maxsize=2)
def _load_shared_template(
    path_value: str,
) -> tuple[np.ndarray, np.ndarray, ShellTopology, str]:
    path = Path(path_value)
    with np.load(path, allow_pickle=False) as payload:
        vertices = np.asarray(payload["vertices"], dtype=np.float32).copy()
        faces = np.asarray(payload["faces"], dtype=np.int64).copy()
        vertex_surface = np.asarray(payload["vertex_surface"], dtype=np.int64).copy()
        face_surface = np.asarray(payload["face_surface"], dtype=np.int64).copy()
        signature = str(np.asarray(payload["signature"]).item())
        normalization = str(np.asarray(payload["normalization"]).item())
    if normalization != SHARED_TEMPLATE_NORMALIZATION:
        raise ValueError(
            f"template normalization={normalization!r}, "
            f"expected={SHARED_TEMPLATE_NORMALIZATION!r}"
        )
    topology = ShellTopology(
        faces=faces,
        epi_vertices=np.flatnonzero(vertex_surface == 0),
        endo_vertices=np.flatnonzero(vertex_surface == 1),
        epi_faces=faces[face_surface == 0],
        endo_faces=faces[face_surface == 1],
        vertex_surface=vertex_surface,
        face_surface=face_surface,
    )
    if len(topology.epi_vertices) != 12836 or len(topology.endo_vertices) != 9207:
        raise ValueError("unexpected shared LV shell topology")
    return vertices, faces, topology, signature


@lru_cache(maxsize=8)
def _load_shared_template_tensors(
    path_value: str, device_value: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize the fixed shell once per process/device.

    Every static case uses the same normalized template. Rebuilding its 22k
    vertices and faces through Python lists for every case was unnecessary CPU
    work and also prevented inexpensive case preloading.
    """

    vertices, faces, _topology, _signature = _load_shared_template(path_value)
    device = torch.device(device_value)
    return (
        torch.tensor(vertices.tolist(), dtype=torch.float32, device=device),
        torch.tensor(faces.tolist(), dtype=torch.long, device=device),
    )


def load_case(
    record: dict[str, Any], config: dict[str, Any], device: torch.device
) -> StaticEndpointCase:
    case_id = str(record["case_id"]).zfill(3)
    path = static_cache_path(config, case_id)
    if not path.exists():
        raise FileNotFoundError(
            f"case {case_id} static cache missing; run scripts/01_prepare_static_case.py"
        )
    with np.load(path, allow_pickle=False) as payload:
        fields = set(payload.files)
        forbidden = sorted(fields & FORBIDDEN_CACHE_FIELDS)
        if forbidden:
            raise ValueError(f"static cache contains forbidden fields: {forbidden}")
        metadata = json.loads(str(np.asarray(payload["metadata"]).item()))
        if metadata.get("version") != STATIC_CACHE_VERSION:
            raise ValueError(f"stale static cache for case {case_id}: {path}")
        arrays = {key: np.asarray(payload[key]) for key in payload.files if key != "metadata"}
    template_path = str(Path(config["data"]["shared_template"]).resolve())
    _vertices, _faces, topology, _signature = _load_shared_template(template_path)
    shared_vertices, shared_faces = _load_shared_template_tensors(
        template_path, str(device)
    )
    def tensor(value: Any, dtype: torch.dtype = torch.float32) -> torch.Tensor:
        # The cluster currently pairs a NumPy-2 runtime with a PyTorch wheel
        # built against NumPy-1. Explicit lists avoid that binary ABI bridge.
        if isinstance(value, np.ndarray):
            value = value.tolist()
        return torch.tensor(value, dtype=dtype, device=device)
    return StaticEndpointCase(
        case_id=case_id,
        split=str(record["split"]),
        contours=tensor(arrays["contours"]),
        contour_valid=tensor(arrays["contour_valid"], torch.bool),
        contour_closed=tensor(arrays["contour_closed"], torch.bool),
        view_ids=tensor(arrays["view_ids"], torch.long),
        slice_positions=tensor(arrays["slice_positions"]),
        plane_origins=tensor(arrays["plane_origins"]),
        plane_normals=tensor(arrays["plane_normals"]),
        plane_u=tensor(arrays["plane_u"]),
        plane_v=tensor(arrays["plane_v"]),
        training_myo_views=tensor(arrays["training_myo_views"]),
        endpoint_cavity_volume_ml=tensor(record["endpoint_cavity_volume_ml"]),
        initial_vertices=shared_vertices,
        faces=shared_faces,
        topology=topology,
        center=np.asarray(arrays["center"], dtype=np.float32),
        scale=float(np.asarray(arrays["scale"]).item()),
        canonical_transform=record["canonical_transform"],
        orientation=np.asarray(record["case_orientation"], dtype=np.float32),
    )


def normalized_to_world(case: StaticEndpointCase, vertices: np.ndarray) -> np.ndarray:
    oriented = np.asarray(vertices, dtype=np.float64) * float(case.scale) + case.center
    canonical = oriented @ np.asarray(case.orientation, dtype=np.float64)
    axes = np.asarray(case.canonical_transform["axes_world"], dtype=np.float64)
    origin = np.asarray(case.canonical_transform["origin_world"], dtype=np.float64)
    return (canonical @ axes + origin).astype(np.float32)
