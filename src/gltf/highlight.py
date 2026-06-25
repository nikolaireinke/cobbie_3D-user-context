"""Resolve agent-emitted highlight GUIDs to GlobalIds the Unity client can draw.

The Unity client highlights an element by looking its GlobalId up among the glTF
nodes it loaded. A glTF node — and therefore a drawable element — exists ONLY for
geometry-bearing IFC elements (see ``src/gltf/convert.py``: the geometry iterator
yields one node per shape, named by GlobalId). Geometry-less assembly containers
have no node: an ``IfcStair`` carries no mesh of its own — the geometry lives on
its ``IfcStairFlight``/``IfcRailing``/``IfcMember`` children — so emitting the
stair's GlobalId highlights nothing, even though the stair exists in the IFC model.

This module bridges that gap: for any emitted GUID that has no glTF node, it
expands the element's decomposition (recursive ``IsDecomposedBy`` — aggregation
and nesting, NOT spatial containment) down to its geometry-bearing leaves and
emits those instead. Geometry-bearing GUIDs (doors, slabs) pass through unchanged.

The "is this drawable?" test reads the node names straight out of the built
``.glb`` — the very artifact Unity resolves against — so the two can't drift.
"""

import json
import os
import struct

_GLB_MAGIC = 0x46546C67  # "glTF" little-endian
_JSON_CHUNK = 0x4E4F534A  # "JSON" little-endian

# Drawable GlobalId sets, cached by (glb_path -> (mtime, guids)). A rebuilt .glb
# (newer mtime) invalidates its entry so a re-export is always picked up.
_cache: dict[str, tuple[float, frozenset[str]]] = {}


def _read_node_names(glb_path: str) -> frozenset[str]:
    """glTF node names from a .glb's JSON chunk (the first chunk). Empty on any
    malformed/unexpected structure — callers treat empty as 'unknown'."""
    with open(glb_path, "rb") as fh:
        header = fh.read(12)
        if len(header) < 12:
            return frozenset()
        magic, _ver, _len = struct.unpack("<III", header)
        if magic != _GLB_MAGIC:
            return frozenset()
        chunk_header = fh.read(8)
        if len(chunk_header) < 8:
            return frozenset()
        clen, ctype = struct.unpack("<II", chunk_header)
        if ctype != _JSON_CHUNK:
            return frozenset()
        chunk = fh.read(clen)
    gltf = json.loads(chunk)
    return frozenset(
        n["name"] for n in gltf.get("nodes", []) if isinstance(n, dict) and n.get("name")
    )


def glb_geometry_guids(glb_path: str) -> frozenset[str]:
    """GlobalIds with geometry in the built .glb (== its glTF node names) — i.e.
    exactly what the Unity client can resolve and highlight. Empty if the .glb is
    missing or unreadable. Cached by (path, mtime)."""
    try:
        mtime = os.path.getmtime(glb_path)
    except OSError:
        return frozenset()
    cached = _cache.get(glb_path)
    if cached is not None and cached[0] == mtime:
        return cached[1]
    try:
        names = _read_node_names(glb_path)
    except Exception:  # noqa: BLE001 - any parse failure -> unknown, pass through
        names = frozenset()
    _cache[glb_path] = (mtime, names)
    return names


def _collect_decomposition_leaves(
    element, drawable: frozenset[str], out: list[str], seen: set[str]
) -> None:
    """Append geometry-bearing GlobalIds found by walking ``element``'s
    decomposition (recursive IsDecomposedBy: IfcRelAggregates + IfcRelNests).
    Order-preserving, deduped via ``seen``."""
    for rel in getattr(element, "IsDecomposedBy", None) or []:
        for child in getattr(rel, "RelatedObjects", None) or []:
            gid = getattr(child, "GlobalId", None)
            if gid and gid in drawable and gid not in seen:
                seen.add(gid)
                out.append(gid)
            _collect_decomposition_leaves(child, drawable, out, seen)


def resolve_highlight_guids(
    model, guids, glb_path: str, cap: int = 500
) -> list[str]:
    """Map emitted highlight GUIDs to drawable (geometry-bearing) GlobalIds.

    For each GUID: keep it if it has a glTF node; otherwise expand its
    decomposition to geometry-bearing leaves and keep those. Order-preserving
    dedupe, capped. If the .glb's drawable set is unknown (missing/unreadable),
    GUIDs pass through unchanged — no .glb means Unity has no geometry to resolve
    against anyway, so there is nothing to expand toward.
    """
    drawable = glb_geometry_guids(glb_path)
    if not drawable:  # unknown drawable set -> don't second-guess the agent
        return list(dict.fromkeys(guids))[:cap]

    out: list[str] = []
    seen: set[str] = set()
    for gid in guids:
        if gid in drawable:
            if gid not in seen:
                seen.add(gid)
                out.append(gid)
            continue
        # No node for this GUID: a geometry-less container. Expand it.
        try:
            element = model.by_guid(gid) if model is not None else None
        except Exception:  # noqa: BLE001 - unknown/malformed guid
            element = None
        if element is not None:
            _collect_decomposition_leaves(element, drawable, out, seen)
    return out[:cap]
