"""Streamlit control panel for the Cobbie WebSocket server.

Replaces the hand-typed launch command with a small UI that configures the
existing `src.server` CLI, starts/stops it, and tails its log live — handy now
that the frontend is an Android VR headset connecting over the LAN.

Launch with:  uv run streamlit run scripts/server_control_app.py

The panel only *drives* `src/server.py`; it spawns it as a subprocess and
supervises it. Behaviour notes:
  - Closing the browser *tab* leaves the server running (the supervisor is a
    process-wide singleton that survives Streamlit reruns/sessions). Stop is
    always explicit via the Stop button.
  - Killing the *Streamlit process* (Ctrl-C in its terminal) stops the server
    too: it is spawned with `--parent-pid <streamlit pid>`, and the server's
    own watchdog self-exits when that parent dies — so it never orphans and
    squats on the port.
"""

import os
import re
import socket
import subprocess
import sys
import threading
import urllib.request
from collections import deque
from contextlib import suppress
from pathlib import Path

import streamlit as st

# Repo root, derived from this file (scripts/ is one level under root) so the
# panel works regardless of the cwd it was launched from.
REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENTS_BAML = REPO_ROOT / "src" / "baml" / "baml_src" / "clients.baml"
DEFAULT_MODEL = str(REPO_ROOT / "src" / "db" / "bim_models" / "duplex" / "arc.ifc")
TOOL_CHOICES = ["initial", "created", "manual"]
FALLBACK_CLIENTS = ["Claude_Haiku_4_5", "Claude_Sonnet_4_6"]


# --------------------------------------------------------------------------
# Server subprocess supervisor (a process-wide singleton via cache_resource).
# --------------------------------------------------------------------------


class ServerSupervisor:
    """Owns the spawned `src.server` process. A daemon thread drains its merged
    stdout/stderr into a bounded deque, parsing the `READY <port>` handshake so
    the UI can show when it is actually serving."""

    def __init__(self) -> None:
        self.proc: subprocess.Popen[str] | None = None
        self.log: deque[str] = deque(maxlen=2000)
        self.ready_port: int | None = None
        self.host: str | None = None
        self.port: int | None = None
        self._reader: threading.Thread | None = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self.proc is not None and self.proc.poll() is None

    def exit_code(self) -> int | None:
        """None while running or never started, else the process exit code."""
        return self.proc.poll() if self.proc is not None else None

    def start(self, cmd: list[str], env: dict[str, str], cwd: str, host: str, port: int) -> None:
        with self._lock:
            if self.is_running():
                return
            self.log.clear()
            self.ready_port = None
            self.host, self.port = host, port
            self.log.append(f"$ {' '.join(cmd)}\n")
            self.proc = subprocess.Popen(
                cmd, cwd=cwd, env=env,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            self._reader = threading.Thread(
                target=self._read_loop, args=(self.proc,), daemon=True, name="server-log-reader"
            )
            self._reader.start()

    def stop(self) -> None:
        with self._lock:
            if self.proc is None or self.proc.poll() is not None:
                return
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                with suppress(subprocess.TimeoutExpired):
                    self.proc.wait(timeout=5)
            self.log.append("\n[control] server stopped.\n")

    def log_text(self, max_lines: int = 800) -> str:
        return "".join(list(self.log)[-max_lines:])

    def _read_loop(self, proc: subprocess.Popen[str]) -> None:
        if proc.stdout is None:
            return
        for line in iter(proc.stdout.readline, ""):
            self.log.append(line)
            stripped = line.strip()
            if stripped.startswith("READY "):
                with suppress(ValueError, IndexError):
                    self.ready_port = int(stripped.split()[1])
        # readline returned "" -> EOF -> the process has exited.


@st.cache_resource
def get_supervisor() -> ServerSupervisor:
    return ServerSupervisor()


# --------------------------------------------------------------------------
# Helpers.
# --------------------------------------------------------------------------


@st.cache_data
def client_names() -> list[str]:
    """BAML client names, parsed from clients.baml so the dropdown stays in sync
    as clients are added/removed. Falls back to a small static list on error."""
    try:
        text = CLIENTS_BAML.read_text(encoding="utf-8")
        names = re.findall(r"client<llm>\s+([A-Za-z0-9_]+)", text)
        return names or FALLBACK_CLIENTS
    except OSError:
        return FALLBACK_CLIENTS


def lan_ip() -> str:
    """Primary LAN IP (the address the headset should target). The UDP-socket
    trick reads the routing choice without sending anything; falls back to
    loopback when offline."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
        finally:
            s.close()
    except OSError:
        return "127.0.0.1"


def check_health(url: str, timeout: float = 2.0) -> tuple[int | None, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:  # noqa: S310 - fixed localhost URL
            return r.status, r.read().decode("utf-8", "replace")[:200]
    except Exception as e:  # noqa: BLE001 - any failure is just a red light
        return None, str(e)


def build_command(
    host: str, port: int, client: str, tools: list[str],
    max_iterations: int, max_concurrency: int, create_gltf: bool,
    model: str, log_dir: str, no_query_logs: bool, no_cache_models: bool, mlflow_dir: str,
) -> list[str]:
    """The exact argv passed to `src.server`. `-u` keeps the child unbuffered so
    its log reaches the pane promptly; `--parent-pid` ties its lifetime to this
    Streamlit process (see module docstring)."""
    cmd = [
        sys.executable, "-u", "-m", "src.server",
        "--host", host,
        "--port", str(port),
        "--client", client,
        "--max-iterations", str(max_iterations),
        "--max-concurrency", str(max_concurrency),
        "--parent-pid", str(os.getpid()),
        "--tools", *tools,  # nargs="+"; argparse stops at the next --flag
    ]
    if model:
        cmd += ["--model", model]
    if create_gltf:
        cmd.append("--create-gltf")
    if log_dir:
        cmd += ["--log-dir", log_dir]
    if no_query_logs:
        cmd.append("--no-query-logs")
    if no_cache_models:
        cmd.append("--no-cache-models")
    if mlflow_dir:
        cmd += ["--mlflow-dir", mlflow_dir]
    return cmd


@st.fragment(run_every="1s")
def render_status_and_log(sup: ServerSupervisor) -> None:
    """Status badge + log pane, re-rendered ~once a second without rerunning the
    whole app, so the log tails live and a crash is reflected promptly."""
    if sup.is_running():
        if sup.ready_port is not None:
            st.success(f"● Running — serving on port {sup.ready_port}")
        else:
            st.warning("● Starting… (waiting for READY)")
    else:
        code = sup.exit_code()
        if code is None:
            st.info("○ Stopped")
        else:
            st.error(f"○ Exited (code {code})")
    with st.container(height=420):
        st.code(sup.log_text() or "(no output yet)", language=None)


# --------------------------------------------------------------------------
# Page.
# --------------------------------------------------------------------------

st.set_page_config(page_title="Cobbie Server Control", layout="wide")
sup = get_supervisor()

with st.sidebar:
    st.header("Server configuration")
    network = st.toggle(
        "Accept network connections", value=True,
        help="On = bind 0.0.0.0 (reachable by the VR headset on the LAN). Off = 127.0.0.1 (local only).",
    )
    host = "0.0.0.0" if network else "127.0.0.1"
    port = int(st.number_input("Port", min_value=1, max_value=65535, value=8000, step=1))

    clients = client_names()
    default_idx = clients.index("Claude_Haiku_4_5") if "Claude_Haiku_4_5" in clients else 0
    client = st.selectbox("BAML client", clients, index=default_idx)
    custom_client = st.text_input("Custom client override", "", help="If set, overrides the dropdown.")
    if custom_client.strip():
        client = custom_client.strip()

    tools = st.multiselect("Tools", TOOL_CHOICES, default=["initial"])
    col_a, col_b = st.columns(2)
    max_iterations = int(col_a.number_input("Max iterations", min_value=1, max_value=100, value=8))
    max_concurrency = int(col_b.number_input("Max concurrency", min_value=1, max_value=16, value=2))
    create_gltf = st.checkbox(
        "Regenerate glTF before serving", value=False,
        help="Run --create-gltf: (re)build .glb geometry for changed/new models before going live.",
    )

    with st.expander("Advanced"):
        model = st.text_input("Default model (--model)", DEFAULT_MODEL)
        log_dir = st.text_input("Log dir (--log-dir)", "", help="Blank = default <root>/query_logs.")
        no_query_logs = st.checkbox("Disable query transcripts (--no-query-logs)", value=False)
        no_cache_models = st.checkbox("Re-open IFC per query (--no-cache-models)", value=False)
        mlflow_dir = st.text_input("MLflow dir (--mlflow-dir)", "", help="Blank = default <root>/mlruns.")

st.title("Cobbie Server Control")
st.caption(
    "Configure and run the Cobbie WebSocket server. The browser tab is a remote control — closing "
    "it leaves the server running; stop it explicitly here, or quit this Streamlit process to shut it down."
)

running = sup.is_running()
c_start, c_stop, c_health = st.columns(3)
start_clicked = c_start.button("▶ Start", type="primary", disabled=running, use_container_width=True)
stop_clicked = c_stop.button("■ Stop", disabled=not running, use_container_width=True)
health_clicked = c_health.button("Check /health", disabled=not running, use_container_width=True)

if start_clicked:
    if not tools:
        st.error("Select at least one tool directory.")
    else:
        cmd = build_command(
            host, port, client, tools, max_iterations, max_concurrency,
            create_gltf, model, log_dir, no_query_logs, no_cache_models, mlflow_dir,
        )
        env = {**os.environ, "ROOT_PATH": str(REPO_ROOT), "PYTHONUNBUFFERED": "1"}
        sup.start(cmd, env, str(REPO_ROOT), host, port)
        st.rerun()

if stop_clicked:
    sup.stop()
    st.rerun()

if health_clicked:
    st.session_state["health"] = check_health(f"http://127.0.0.1:{port}/health")

health = st.session_state.get("health")
if health is not None:
    status, body = health
    if status == 200:
        st.success(f"/health 200 — {body}")
    else:
        st.error(f"/health unreachable — {body}")

# Where the headset connects. 0.0.0.0 binds all interfaces, so advertise the LAN
# IP (the headset can't dial 0.0.0.0); a specific host is shown as-is.
advertised = lan_ip() if host == "0.0.0.0" else host
st.markdown(
    f"**Headset endpoints** (same Wi-Fi):  \n"
    f"• WebSocket — `ws://{advertised}:{port}/ws`  \n"
    f"• Models — `http://{advertised}:{port}/models`  \n"
    f"• Health — `http://{advertised}:{port}/health`"
)

st.divider()
render_status_and_log(sup)
