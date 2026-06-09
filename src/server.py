# src/server.py
#
# Local/networked WebSocket server that exposes the cobbie BIM agent to the
# Unity client. Pairs with BackendLauncher.cs:
#
#   - Launched as:  python -m src.server --port <P> [--parent-pid <PID>]
#   - Binds the port, then prints exactly "READY <port>" to stdout once the app
#     is ready (tools loaded, imports done). The launcher reads that line.
#   - GET  /health  -> 200 once serving (used by the Remote provider).
#   - GET  /models  -> catalogue of registered IFC models (for a client-side picker).
#   - GET  /model-gltf?model_id=<int> -> stream that model's .glb (nodes named by
#                       IFC GlobalId). Requires the glb to have been generated
#                       (server --create-gltf, or `python -m src.gltf.convert`).
#   - WS   /ws       -> query channel (see protocol below).
#
# WebSocket protocol (v2):
#   client -> {"type":"query", "question": str, "context": {...}?,
#              "model_id": int?, "model_path": str?}
#       Model selection precedence: model_id (DB lookup) > model_path > --model default.
#       context: {"selection":[guid,...], "objects_in_view":[{...}], "user_pose":{...}}
#   server -> {"type":"status", "stage":"running", "model": str}
#   server -> {"type":"iteration", "iteration": int, "thoughts": str, "code": str}   (streamed)
#   server -> {"type":"observation", "iteration": int, "result": str}                (streamed)
#   server -> {"type":"highlight", "guids":[guid,...]}   (GlobalIds for the viewer
#              to highlight; emitted once, non-empty, just before the final message.
#              Resolved server-side to geometry-bearing GlobalIds the client can
#              draw — geometry-less containers like IfcStair are expanded to their
#              geometry leaves; suppressed entirely if nothing is drawable.)
#   server -> {"type":"final", "answer": str, "reasoning": str, "success": bool}
#         or  {"type":"error", "error": str, "error_type": str}
#
# DEFERRED (lands with the inner_loop pass):
#   - Per-iteration streaming of thoughts/code/results instead of one final msg.
#   - Real context injection (binding selection/pose as code-prefix variables);
#     for now context is folded into the prompt as an interim bridge.
#   - Caching the opened ifcopenshell model across queries.
#   - Cooperative cancellation when the client disconnects mid-query.

import argparse
import asyncio
import datetime
import json
import os
import socket
import sys
import threading
import time
from contextlib import suppress
from typing import Any

import mlflow
import uvicorn
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse

from src.agents.cobbie import cobbie
from src.config import TEST_IFC_PATH, DIRECTORY_IFC_MODELS_PATH
from src.db.query import get_ifc_model, get_ifc_models
from src.gltf.convert import build_all, glb_path_for, glb_path_from_ifc
from src.gltf.highlight import resolve_highlight_guids
from src.util.get_tools import get_tools

# --------------------------------------------------------------------------
# Runtime config, populated in main().
# --------------------------------------------------------------------------


class ServerState:
    tools: dict = {}
    default_model_path: str = TEST_IFC_PATH
    client: str = "Claude_Sonnet_4_6"
    max_iterations: int = 5
    add_code_prefix: bool = True  # binds path_ifc_model in the per-iteration prefix
    cache_models: bool = True     # keep opened ifcopenshell.file objects across queries
    log_dir: str = None           # per-query transcript dir; None disables


STATE = ServerState()

# Opened IFC files, keyed by resolved path. ifcopenshell.file is not safe for
# concurrent access, so _AGENT_LOCK serializes agent runs while caching is on.
# Fine for single-user research; revisit (per-model locks / per-request copies)
# before serving multiple concurrent users.
_MODEL_CACHE: dict = {}
_AGENT_LOCK = threading.Lock()


def get_cached_ifc(path: str):
    model = _MODEL_CACHE.get(path)
    if model is None:
        import ifcopenshell  # already a dependency; imported lazily
        model = ifcopenshell.open(path)
        _MODEL_CACHE[path] = model
    return model


def resolve_drawable_highlights(guids: list[str], model_path: str) -> list[str]:
    """Map agent-emitted highlight GUIDs to GlobalIds the Unity client can
    actually draw (geometry-bearing glTF nodes), expanding geometry-less
    assembly containers to their geometry leaves. Best-effort: any failure
    returns the original list so a resolver bug never silently drops a highlight
    the client would otherwise have shown. Runs on the agent's worker thread, so
    the cached ifcopenshell model is accessed single-threaded."""
    if not guids:
        return guids
    try:
        model = get_cached_ifc(model_path)
        return resolve_highlight_guids(model, guids, glb_path_from_ifc(model_path))
    except Exception as e:  # noqa: BLE001 - never let highlight resolution break the run
        print(f"[server] highlight resolution failed: {e}", file=sys.stderr, flush=True)
        return guids


def log_highlight(f, original: list[str], resolved: list[str]) -> None:
    """Transcript line for a highlight, showing the original -> drawable mapping
    so 'emitted N, drew M' bugs are diagnosable at a glance."""
    if f is None:
        return
    if resolved == original:
        f.write(f"\n[highlight] {len(resolved)} object(s): {', '.join(resolved)}\n")
    elif resolved:
        f.write(
            f"\n[highlight] {len(original)} emitted -> {len(resolved)} drawable: "
            f"{', '.join(resolved)}\n"
        )
    else:
        f.write(
            f"\n[highlight] {len(original)} emitted -> 0 drawable (suppressed): "
            f"{', '.join(original)}\n"
        )
    f.flush()


# --------------------------------------------------------------------------
# Per-query transcript logging (full, untruncated — unlike the console view).
# --------------------------------------------------------------------------


def open_query_log(question: str, model_path: str, context: dict | None = None):
    """Open a per-query transcript file. Returns (file, path) or (None, None)."""
    if not STATE.log_dir:
        return None, None
    os.makedirs(STATE.log_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() else "_" for c in question[:40]).strip("_") or "query"
    path = os.path.join(STATE.log_dir, f"{ts}_{safe}.log")
    f = open(path, "w", encoding="utf-8")
    f.write(f"timestamp : {datetime.datetime.now().isoformat()}\n")
    f.write(f"model     : {model_path}\n")
    f.write(f"question  : {question}\n")
    if context:
        sel = context.get("selection") or []
        view = context.get("objects_in_view") or []
        pose = context.get("user_pose") or {}
        f.write(f"selection : {sel}\n")
        f.write(f"in_view   : {len(view)} object(s)\n")
        f.write(f"user_pose : {pose}\n")
        f.write("context (full json):\n")
        f.write(json.dumps(context, ensure_ascii=False, indent=2) + "\n")
    else:
        f.write("context   : (none)\n")
    f.write("=" * 80 + "\n")
    f.flush()
    return f, path


def log_event(f, ev: dict):
    """Append one streamed event in full to the transcript (called on the worker thread)."""
    if f is None:
        return
    t = ev.get("type")
    if t == "iteration":
        ll = ev.get("llm_seconds")
        header = f"\n----- ITERATION {ev.get('iteration')} -----"
        if ll is not None:
            header += f"  (llm {round(ll, 3)}s)"
        f.write(header + "\n")
        f.write(f"[thoughts]\n{ev.get('thoughts', '')}\n\n")
        f.write(f"[code]\n{ev.get('code', '')}\n")
    elif t == "observation":
        ex = ev.get("exec_seconds")
        suffix = f"  (exec {ex}s)" if ex is not None else ""
        f.write(f"\n[result of iteration {ev.get('iteration')}]{suffix}\n{ev.get('result', '')}\n")
    # "highlight" events are logged via log_highlight (original -> drawable);
    # "timing" events are folded into the end-of-query TIMING SUMMARY block.
    f.flush()


def write_timing_summary(f, t: dict):
    """Write the end-of-query timing breakdown to the transcript."""
    if f is None:
        return
    f.write("\n" + "-" * 80 + "\n")
    f.write("TIMING SUMMARY (seconds)\n")
    f.write(f"  total               : {t['total_s']}\n")
    f.write(f"  time to first token : {t['time_to_first_token_s']}\n")
    f.write(f"  model load          : {t['model_load_s']}\n")
    f.write(f"  LLM (all calls)     : {t['llm_s']}\n")
    f.write(f"  code execution      : {t['exec_s']}\n")
    f.write(f"  residual / overhead : {t['residual_s']}\n")
    f.write(
        f"  iterations={t['iterations']} llm_calls={t['llm_calls']} "
        f"code_executions={t['code_executions']}\n"
    )
    f.flush()
app = FastAPI(title="cobbie-bridge")

# Allow browser/WebGL builds (which enforce CORS, unlike the Editor/Standalone
# UnityWebRequest) to fetch /models and the .glb geometry cross-origin.
# Permissive by design: a local single-user research bridge serving GET-only
# data with no credentials. Tighten allow_origins if ever exposed beyond
# localhost. expose_headers lets the client read the range/length headers
# glTFast may rely on when streaming the .glb.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Length", "Content-Range", "Accept-Ranges", "Content-Disposition"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "model": STATE.default_model_path}


@app.get("/models")
async def list_models():
    """Catalogue the registered IFC models so the client can show a picker.
    The client then sends the chosen `model_id` with each query."""
    models = await asyncio.to_thread(get_ifc_models)
    return {
        "models": [
            {
                "id": m.id,
                "project_name": m.project_name,
                "model_name": m.model_name,
                "description": m.model_description,
            }
            for m in models
        ]
    }


@app.get("/elements")
async def list_elements(
    model_id: int | None = None,
    model_path: str | None = None,
    ifc_class: str = "IfcElement",
    limit: int = 1000,
):
    """List elements (GlobalId, class, name) of a model so a client can offer a
    selection picker with IDs that model.by_guid() is guaranteed to resolve."""
    try:
        path = resolve_model(model_id, model_path)
    except ModelResolutionError as e:
        raise HTTPException(status_code=404, detail=str(e))

    def work():
        # Touch the (possibly shared) cached file under the agent lock.
        with _AGENT_LOCK:
            if STATE.cache_models:
                model = get_cached_ifc(path)
            else:
                import ifcopenshell
                model = ifcopenshell.open(path)
            try:
                elems = model.by_type(ifc_class)  # includes subclasses
            except RuntimeError as e:
                raise HTTPException(status_code=400, detail=f"Bad ifc_class: {e}")
            return [
                {"guid": e.GlobalId, "ifc_class": e.is_a(), "name": getattr(e, "Name", None)}
                for e in elems[: max(0, limit)]
            ]

    elements = await asyncio.to_thread(work)
    return {"model": path, "count": len(elements), "elements": elements}


def _entity_brief(e) -> dict | None:
    """Compact {ifc_class, name} for a related entity (storey, type, ...)."""
    if e is None:
        return None
    return {"ifc_class": e.is_a(), "name": getattr(e, "Name", None)}


@app.get("/element")
async def get_element(
    guid: str,
    model_id: int | None = None,
    model_path: str | None = None,
):
    """Metadata for one element, looked up by GlobalId — the same id the viewer
    selects and highlights. Feeds a selection-detail panel: identity, spatial
    container, type, materials, and every property/quantity set. Geometry stays
    in the .glb; this serves the semantics on demand from the live IFC model."""
    try:
        path = resolve_model(model_id, model_path)
    except ModelResolutionError as e:
        raise HTTPException(status_code=404, detail=str(e))

    def work():
        import ifcopenshell.util.element as ue

        # Touch the (possibly shared) cached file under the agent lock.
        with _AGENT_LOCK:
            if STATE.cache_models:
                model = get_cached_ifc(path)
            else:
                import ifcopenshell
                model = ifcopenshell.open(path)
            try:
                el = model.by_guid(guid)
            except RuntimeError:
                el = None
            if el is None:
                raise HTTPException(status_code=404, detail=f"No element with guid={guid}")
            return {
                "guid": el.GlobalId,
                "ifc_class": el.is_a(),
                "name": getattr(el, "Name", None),
                "description": getattr(el, "Description", None),
                "tag": getattr(el, "Tag", None),
                "object_type": getattr(el, "ObjectType", None),
                "container": _entity_brief(ue.get_container(el)),
                "type": _entity_brief(ue.get_type(el)),
                "materials": [
                    m.Name for m in ue.get_materials(el) if getattr(m, "Name", None)
                ],
                "psets": ue.get_psets(el),  # property + quantity sets, primitives
            }

    return await asyncio.to_thread(work)


@app.get("/model-gltf")
async def model_gltf(model_id: int):
    """Stream a model's generated .glb (glTF nodes are named by IFC GlobalId).

    The client selects a registered model via /models, then fetches its geometry
    here with the same model_id it sends on queries."""
    try:
        glb_path = resolve_glb(model_id)
    except ModelResolutionError as e:
        raise HTTPException(status_code=404, detail=str(e))
    size_mb = os.path.getsize(glb_path) / (1024 * 1024)
    print(
        f"[server] serving glTF: model_id={model_id} "
        f"{os.path.basename(glb_path)} ({size_mb:.1f} MB)",
        file=sys.stderr,
        flush=True,
    )
    return FileResponse(
        glb_path,
        media_type="model/gltf-binary",
        filename=os.path.basename(glb_path),
    )


# --------------------------------------------------------------------------
# Model resolution.
# --------------------------------------------------------------------------


class ModelResolutionError(Exception):
    def __init__(self, message: str, error_type: str = "ModelNotFound"):
        super().__init__(message)
        self.error_type = error_type


def _validate_ifc(path: str) -> str:
    if not path.lower().endswith(".ifc"):
        raise ModelResolutionError(f"Not an .ifc file: {path}", "BadModel")
    if not os.path.isfile(path):
        raise ModelResolutionError(f"IFC file not found on server: {path}", "ModelNotFound")
    return path


def resolve_model(model_id, model_path) -> str:
    """Resolve a query's model selection to a path on THIS machine.

    Precedence: model_id (DB) > model_path (same-machine dev) > server default.
    The DB stores absolute paths baked to the machine that populated it, so for
    model_id we first try a path re-rooted at this machine's bim_models dir and
    only fall back to the stored absolute path if that file actually exists.
    """
    if model_id is not None:
        rec = get_ifc_model(int(model_id))
        if rec is None:
            raise ModelResolutionError(f"No model registered with id={model_id}", "ModelNotFound")

        rerooted = os.path.join(DIRECTORY_IFC_MODELS_PATH, rec.project_name, f"{rec.model_name}.ifc")
        if os.path.isfile(rerooted):
            return rerooted
        if os.path.isfile(rec.model_path):  # stored absolute path, if it happens to be valid here
            return rec.model_path
        raise ModelResolutionError(
            f"Model id={model_id} ({rec.project_name}/{rec.model_name}) is registered but its "
            f"IFC file is missing on the server. Expected: {rerooted}",
            "ModelNotFound",
        )

    if model_path:
        return _validate_ifc(model_path)

    return _validate_ifc(STATE.default_model_path)


def resolve_glb(model_id) -> str:
    """Resolve a registered model_id to its generated .glb path on this machine.

    Keyed off the same project/model_name as resolve_model so the glb mirrors the
    IFC selection. Raises ModelResolutionError if unregistered or not yet generated.
    """
    if model_id is None:
        raise ModelResolutionError("model_id is required for /model-gltf", "BadRequest")
    rec = get_ifc_model(int(model_id))
    if rec is None:
        raise ModelResolutionError(f"No model registered with id={model_id}", "ModelNotFound")

    glb_path = glb_path_for(rec.project_name, rec.model_name)
    if not os.path.isfile(glb_path):
        raise ModelResolutionError(
            f"glTF for id={model_id} ({rec.project_name}/{rec.model_name}) not generated. "
            f"Run the server with --create-gltf (or `python -m src.gltf.convert`).",
            "GltfNotGenerated",
        )
    return glb_path


# --------------------------------------------------------------------------
# Blocking agent call, run off the event loop.
# --------------------------------------------------------------------------


def run_cobbie_blocking(
    question: str,
    context: dict | None,
    model_path: str,
    on_event=None,
) -> dict[str, Any]:
    # Serialize agent runs while caching (shared ifcopenshell.file isn't
    # concurrency-safe). With caching off, runs are independent.
    with _AGENT_LOCK:
        # Time the (possibly cold) IFC open so the transcript can separate model
        # load from reasoning. Cached hits are ~0; first open can be seconds.
        _load_start = time.time()
        ifc_model = get_cached_ifc(model_path) if STATE.cache_models else None
        model_load_s = round(time.time() - _load_start, 3)
        result = cobbie(
            user_input=question,
            tools=STATE.tools,
            max_iterations=STATE.max_iterations,
            model_path=model_path,
            add_code_prefix=STATE.add_code_prefix,
            client=STATE.client,
            user_context=context,   # injected as model.by_guid-resolvable variables
            ifc_model=ifc_model,    # pre-opened; None -> agent opens from model_path
            on_event=on_event,      # per-iteration streaming
        )

    if result.error is not None:
        return {
            "type": "error",
            "error": getattr(result.error, "error_message", str(result.error)),
            "error_type": getattr(result.error, "error_type", "AgentError"),
            "model_load_s": model_load_s,
        }

    if result.answer is None:
        return {
            "type": "error",
            "error": "No answer produced.",
            "error_type": "EmptyResult",
            "model_load_s": model_load_s,
        }

    answer_text = result.answer.answer or ""
    return {
        "type": "final",
        "answer": answer_text,
        "reasoning": result.answer.thoughts or "",
        "success": "iteration limit" not in answer_text.lower(),
        "model_load_s": model_load_s,
    }


# --------------------------------------------------------------------------
# WebSocket endpoint.
# --------------------------------------------------------------------------


@app.websocket("/ws")
async def ws_endpoint(ws: WebSocket):
    await ws.accept()
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") != "query":
                await ws.send_json(
                    {"type": "error", "error": f"Unknown message type: {msg.get('type')}",
                     "error_type": "BadRequest"}
                )
                continue

            question = (msg.get("question") or "").strip()
            if not question:
                await ws.send_json(
                    {"type": "error", "error": "Empty question.", "error_type": "BadRequest"}
                )
                continue

            context = msg.get("context")
            try:
                model_path = resolve_model(msg.get("model_id"), msg.get("model_path"))
            except ModelResolutionError as e:
                await ws.send_json({"type": "error", "error": str(e), "error_type": e.error_type})
                continue

            # Start the wall-clock for this query (covers model load + reasoning).
            t_start = time.time()
            await ws.send_json({"type": "status", "stage": "running", "model": model_path})

            # Full per-query transcript (untruncated), for inspecting code/results.
            log_f, log_path = open_query_log(question, model_path, context)

            # Bridge: cobbie's on_event fires on the worker thread; hand each
            # event to the event loop, and drain them to the socket concurrently
            # while the blocking run proceeds.
            loop = asyncio.get_running_loop()
            event_q: asyncio.Queue = asyncio.Queue()

            # Timing captured from the event stream: time-to-first-token (first
            # iteration the user sees) and the agent's authoritative totals.
            ttft_holder: dict = {"t": None}
            agent_timing: dict = {}

            def on_event(ev, _loop=loop, _q=event_q, _f=log_f, _mp=model_path):
                etype = ev.get("type")
                if etype == "iteration" and ttft_holder["t"] is None:
                    ttft_holder["t"] = time.time() - t_start
                elif etype == "timing":
                    agent_timing.update(ev)
                elif etype == "highlight":
                    # Map the agent's GUIDs to ones Unity can actually draw
                    # (expanding geometry-less containers to their geometry
                    # leaves) before it hits the wire. The drawable set comes
                    # from the built .glb, so it matches what the client resolves.
                    original = ev.get("guids") or []
                    resolved = resolve_drawable_highlights(original, _mp)
                    log_highlight(_f, original, resolved)
                    if not resolved:  # nothing drawable -> contract says don't emit
                        return
                    ev = {**ev, "guids": resolved}
                    _loop.call_soon_threadsafe(_q.put_nowait, ev)
                    return
                log_event(_f, ev)                       # full transcript (worker thread)
                _loop.call_soon_threadsafe(_q.put_nowait, ev)  # stream to socket

            async def drain():
                while True:
                    ev = await event_q.get()
                    if ev is None:  # sentinel: run finished
                        return
                    await ws.send_json(ev)

            drain_task = asyncio.create_task(drain())
            payload: dict[str, Any]
            try:
                payload = await asyncio.to_thread(
                    run_cobbie_blocking, question, context, model_path, on_event
                )
            except Exception as e:  # noqa: BLE001 - surface any agent failure to the client
                payload = {"type": "error", "error": str(e), "error_type": type(e).__name__}
            finally:
                event_q.put_nowait(None)  # stop the drain task
                await drain_task

            # Compose the timing breakdown: agent totals (llm/exec) plus the
            # numbers only the server sees (model load, total wall-clock, ttft).
            # residual surfaces unaccounted time (interpreter setup, tools docs,
            # span + serialization overhead, thread handoff) so the buckets stay
            # honest — a large residual is itself an optimization target.
            total_s = time.time() - t_start
            model_load_s = float(payload.pop("model_load_s", 0.0) or 0.0)
            llm_s = float(agent_timing.get("llm_s", 0.0) or 0.0)
            exec_s = float(agent_timing.get("exec_s", 0.0) or 0.0)
            ttft_s = ttft_holder["t"] if ttft_holder["t"] is not None else total_s
            timing = {
                "total_s": round(total_s, 3),
                "time_to_first_token_s": round(ttft_s, 3),
                "model_load_s": round(model_load_s, 3),
                "llm_s": round(llm_s, 3),
                "exec_s": round(exec_s, 3),
                "residual_s": round(total_s - model_load_s - llm_s - exec_s, 3),
                "iterations": agent_timing.get("iterations"),
                "llm_calls": agent_timing.get("llm_calls"),
                "code_executions": agent_timing.get("code_executions"),
            }
            payload["timing"] = timing

            if log_f is not None:
                write_timing_summary(log_f, timing)
                log_f.write("\n" + "=" * 80 + "\n")
                log_f.write(f"[{payload.get('type')}]\n{json.dumps(payload, ensure_ascii=False, indent=2)}\n")
                log_f.close()
                print(f"[server] query transcript: {log_path}", file=sys.stderr, flush=True)

            await ws.send_json(payload)

    except WebSocketDisconnect:
        # Note: a query already handed to the thread keeps running until done;
        # true cancellation arrives with the inner_loop changes.
        pass


# --------------------------------------------------------------------------
# Parent-process watchdog: exit if the Unity process dies (e.g. a crash) so the
# backend never orphans. Optional; only armed when --parent-pid is given.
# --------------------------------------------------------------------------


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)  # POSIX existence check; no signal sent
    except OSError:
        return False
    return True


def start_parent_watchdog(parent_pid: int):
    def watch():
        while True:
            time.sleep(2.0)
            if not _pid_alive(parent_pid):
                os._exit(0)

    threading.Thread(target=watch, daemon=True, name="parent-watchdog").start()


# --------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="cobbie WebSocket bridge")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000,
                        help="0 = OS-chosen free port (printed in the READY line)")
    parser.add_argument("--model", default=TEST_IFC_PATH,
                        help="Absolute path to the IFC file the viewer has loaded")
    parser.add_argument("--client", default="Claude_Sonnet_4_6", help="BAML primary client")
    parser.add_argument("--tools", nargs="+", default=["initial"],
                        choices=["initial", "created", "manual"])
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument("--no-cache-models", action="store_true",
                        help="Re-open the IFC on every query instead of caching it per path")
    parser.add_argument("--log-dir", default=None,
                        help="Dir for full per-query transcripts (default: <root>/query_logs)")
    parser.add_argument("--no-query-logs", action="store_true",
                        help="Disable per-query transcript files")
    parser.add_argument("--mlflow-dir", default=None,
                        help="Local MLflow file store (default: <root>/mlruns)")
    parser.add_argument("--create-gltf", action="store_true",
                        help="Convert registered IFC models to .glb (incremental) before serving")
    args = parser.parse_args()

    # Force a local, server-less MLflow store so agent runs never hang trying to
    # reach config.MLFLOW_URI (http://127.0.0.1:5000).
    root = os.environ.get("ROOT_PATH", os.getcwd())
    mlflow_dir = args.mlflow_dir or os.path.join(root, "mlruns")
    mlflow.set_tracking_uri(f"file:{mlflow_dir}")
    mlflow.set_experiment("cobbie-bridge")

    # Load tools once and reuse across queries.
    STATE.tools = get_tools(directories=args.tools)
    STATE.default_model_path = args.model
    STATE.client = args.client
    STATE.max_iterations = args.max_iterations
    STATE.cache_models = not args.no_cache_models
    STATE.log_dir = None if args.no_query_logs else (args.log_dir or os.path.join(root, "query_logs"))
    print(f"[server] loaded {len(STATE.tools)} tools; model={args.model}", file=sys.stderr, flush=True)

    # Incrementally (re)generate model geometry before going live. Up-to-date glbs
    # are skipped via a fast mtime check; only changed/new models reconvert.
    if args.create_gltf:
        print("[server] --create-gltf: checking model geometry...", file=sys.stderr, flush=True)
        summary = build_all()
        print(f"[server] glTF: {summary}", file=sys.stderr, flush=True)

    if args.parent_pid:
        start_parent_watchdog(args.parent_pid)

    # Bind first so we can report the actual port (handles --port 0), then serve.
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((args.host, args.port))
    actual_port = sock.getsockname()[1]

    # The launcher waits for exactly this line on stdout.
    print(f"READY {actual_port}", flush=True)

    config = uvicorn.Config(app, log_level="info", access_log=False)
    server = uvicorn.Server(config)
    with suppress(KeyboardInterrupt):
        asyncio.run(server.serve(sockets=[sock]))


if __name__ == "__main__":
    main()
