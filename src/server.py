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
#   server -> {"type":"final", "answer": str, "reasoning": str, "success": bool}
#         or  {"type":"error", "error": str, "error_type": str}
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
import json
import os
import socket
import sys
import threading
import time
from contextlib import suppress

import mlflow
import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from src.agents.cobbie import cobbie
from src.config import TEST_IFC_PATH, DIRECTORY_IFC_MODELS_PATH
from src.db.query import get_ifc_model, get_ifc_models
from src.util.get_tools import get_tools

# --------------------------------------------------------------------------
# Runtime config, populated in main().
# --------------------------------------------------------------------------


class ServerState:
    tools: dict = {}
    default_model_path: str = TEST_IFC_PATH
    client: str = "GLM_4_7"
    max_iterations: int = 15
    add_code_prefix: bool = True  # binds path_ifc_model in the per-iteration prefix
    cache_models: bool = True     # keep opened ifcopenshell.file objects across queries


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
app = FastAPI(title="cobbie-bridge")


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


# --------------------------------------------------------------------------
# Blocking agent call, run off the event loop.
# --------------------------------------------------------------------------


def run_cobbie_blocking(
    question: str,
    context: dict | None,
    model_path: str,
    on_event=None,
) -> dict:
    # Serialize agent runs while caching (shared ifcopenshell.file isn't
    # concurrency-safe). With caching off, runs are independent.
    with _AGENT_LOCK:
        ifc_model = get_cached_ifc(model_path) if STATE.cache_models else None
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
        }

    if result.answer is None:
        return {"type": "error", "error": "No answer produced.", "error_type": "EmptyResult"}

    answer_text = result.answer.answer or ""
    return {
        "type": "final",
        "answer": answer_text,
        "reasoning": result.answer.thoughts or "",
        "success": "iteration limit" not in answer_text.lower(),
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

            await ws.send_json({"type": "status", "stage": "running", "model": model_path})

            # Bridge: cobbie's on_event fires on the worker thread; hand each
            # event to the event loop, and drain them to the socket concurrently
            # while the blocking run proceeds.
            loop = asyncio.get_running_loop()
            event_q: asyncio.Queue = asyncio.Queue()

            def on_event(ev, _loop=loop, _q=event_q):
                _loop.call_soon_threadsafe(_q.put_nowait, ev)

            async def drain():
                while True:
                    ev = await event_q.get()
                    if ev is None:  # sentinel: run finished
                        return
                    await ws.send_json(ev)

            drain_task = asyncio.create_task(drain())
            try:
                payload = await asyncio.to_thread(
                    run_cobbie_blocking, question, context, model_path, on_event
                )
            except Exception as e:  # noqa: BLE001 - surface any agent failure to the client
                payload = {"type": "error", "error": str(e), "error_type": type(e).__name__}
            finally:
                event_q.put_nowait(None)  # stop the drain task
                await drain_task

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
    parser.add_argument("--client", default="GLM_4_7", help="BAML primary client")
    parser.add_argument("--tools", nargs="+", default=["initial"],
                        choices=["initial", "created", "manual"])
    parser.add_argument("--max-iterations", type=int, default=15)
    parser.add_argument("--parent-pid", type=int, default=None)
    parser.add_argument("--no-cache-models", action="store_true",
                        help="Re-open the IFC on every query instead of caching it per path")
    parser.add_argument("--mlflow-dir", default=None,
                        help="Local MLflow file store (default: <root>/mlruns)")
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
    print(f"[server] loaded {len(STATE.tools)} tools; model={args.model}", file=sys.stderr, flush=True)

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
