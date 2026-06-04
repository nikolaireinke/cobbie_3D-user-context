"""IFC -> glTF (.glb) conversion for the Unity client.

Each registered IFC model is converted once to a single-file ``.glb`` whose glTF
node names are the IFC ``GlobalId`` strings (``use-element-guids``). That gives the
roundtrip IFC element -> glTF node -> Unity GameObject -> same GlobalId, which the
Unity-side two-way selection registry depends on.

Conversion is incremental: a model is (re)converted only when its ``.glb`` is missing
or older than the source ``.ifc``. Output mirrors the model layout:
``src/db/bim_models/<project>/<model>.ifc`` -> ``src/db/gltf_models/<project>/<model>.glb``.

Run standalone with ``uv run python -m src.gltf.convert [--force]`` or from the server
via ``--create-gltf`` (see ``src/server.py``).

Note: the native serializer emits a *flat* node list (all products at the scene root);
spatial nesting (Project/Site/Building/Storey) is a deferred follow-up.
"""

import os
import time

import ifcopenshell
import ifcopenshell.geom as geom
from loguru import logger

from src.config import DIRECTORY_GLTF_MODELS_PATH, DIRECTORY_IFC_MODELS_PATH
from src.db.query import get_ifc_models


def glb_path_for(project_name: str, model_name: str) -> str:
    """Output ``.glb`` path mirroring the IFC model layout under bim_models/."""
    return os.path.join(DIRECTORY_GLTF_MODELS_PATH, project_name, f"{model_name}.glb")


def ifc_path_for(rec) -> str | None:
    """Resolve a registered model's IFC path on this machine.

    Mirrors ``resolve_model`` in src/server.py: prefer the path re-rooted at this
    machine's bim_models dir, fall back to the stored absolute path, else None.
    """
    rerooted = os.path.join(DIRECTORY_IFC_MODELS_PATH, rec.project_name, f"{rec.model_name}.ifc")
    if os.path.isfile(rerooted):
        return rerooted
    if os.path.isfile(rec.model_path):
        return rec.model_path
    return None


def is_stale(ifc_path: str, glb_path: str) -> bool:
    """True if the glb is missing or older than the source IFC."""
    if not os.path.isfile(glb_path):
        return True
    return os.path.getmtime(ifc_path) > os.path.getmtime(glb_path)


def ifc_to_glb(ifc_path: str, out_path: str, num_threads: int | None = None) -> int:
    """Convert one IFC file to a GlobalId-named ``.glb``. Returns element count.

    Writes to a temp file then atomically replaces the target, so an interrupted
    run never leaves a half-written ``.glb`` that ``is_stale`` would treat as fresh.
    """
    num_threads = num_threads or (os.cpu_count() or 1)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    f = ifcopenshell.open(ifc_path)

    gs = geom.settings()
    gs.set("apply-default-materials", True)

    ss = geom.serializer_settings()
    ss.set("use-element-guids", True)  # glTF node name == IFC GlobalId
    ss.set("y-up", True)               # glTF/Unity Y-up convention

    tmp_path = f"{out_path}.tmp"
    serializer = geom.serializers.gltf(tmp_path, gs, ss)
    serializer.setFile(f)
    serializer.writeHeader()

    count = 0
    iterator = geom.iterator(gs, f, num_threads)
    if iterator.initialize():
        while True:
            serializer.write(iterator.get())
            count += 1
            if not iterator.next():
                break
    serializer.finalize()

    os.replace(tmp_path, out_path)
    return count


def build_all(force: bool = False) -> dict:
    """Convert all DB-registered models whose glb is missing or stale.

    Per-model failures are caught and reported so one bad model never aborts the
    pass. Returns counts: ``{converted, skipped, missing, failed}``.
    """
    summary = {"converted": 0, "skipped": 0, "missing": 0, "failed": 0}
    models = get_ifc_models()
    logger.info(f"glTF build: {len(models)} registered model(s); force={force}")

    for rec in models:
        tag = f"{rec.project_name}/{rec.model_name}"
        ifc_path = ifc_path_for(rec)
        if ifc_path is None:
            logger.warning(f"  [missing] {tag}: IFC file not found on this machine")
            summary["missing"] += 1
            continue

        glb_path = glb_path_for(rec.project_name, rec.model_name)
        if not force and not is_stale(ifc_path, glb_path):
            summary["skipped"] += 1
            continue

        try:
            t0 = time.time()
            n = ifc_to_glb(ifc_path, glb_path)
            logger.info(f"  [ok] {tag}: {n} elements in {time.time() - t0:.1f}s")
            summary["converted"] += 1
        except Exception as e:  # noqa: BLE001 - keep converting the rest
            logger.error(f"  [fail] {tag}: {e}")
            summary["failed"] += 1

    logger.info(
        "glTF build done: "
        f"converted={summary['converted']} skipped={summary['skipped']} "
        f"missing={summary['missing']} failed={summary['failed']}"
    )
    return summary


if __name__ == "__main__":
    import argparse

    from src.util.setup_logger import setup_logger

    parser = argparse.ArgumentParser(description="Convert registered IFC models to glTF (.glb)")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Reconvert every model even if its glb is already up to date",
    )
    args = parser.parse_args()

    setup_logger()
    summary = build_all(force=args.force)
    print(summary)
