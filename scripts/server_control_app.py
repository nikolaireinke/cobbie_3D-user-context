"""Streamlit control panel for the Cobbie WebSocket server.

Replaces the hand-typed launch command with a small UI that configures the
existing `src.server` CLI, starts/stops it, and tails its log live — handy now
that the frontend is an Android VR headset connecting over the LAN.

Launch with:  uv run streamlit run scripts/server_control_app.py

The panel only *drives* `src/server.py`; it spawns it as a subprocess and
supervises it. Behaviour notes:
  - Closing the browser *tab* stops the server too, while the "Stop server when
    I close this tab" toggle is on (the default): a supervisor watchdog polls
    Streamlit's active-session count and shuts the server down once no tab has
    been connected for TAB_CLOSE_GRACE_SECONDS. The grace window lets a page
    refresh (which briefly disconnects, then reconnects) pass without a kill,
    and a backgrounded tab keeps its websocket open so it never false-fires. A
    `beforeunload` confirm dialog guards against closing the tab by accident.
    Untick the toggle to keep the server running after the tab closes (handy to
    keep the headset served while you step away from the laptop).
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
import time
import urllib.request
from collections import deque
from contextlib import suppress
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

# Repo root, derived from this file (scripts/ is one level under root) so the
# panel works regardless of the cwd it was launched from.
REPO_ROOT = Path(__file__).resolve().parents[1]
CLIENTS_BAML = REPO_ROOT / "src" / "baml" / "baml_src" / "clients.baml"
DEFAULT_MODEL = str(REPO_ROOT / "src" / "db" / "bim_models" / "duplex" / "arc.ifc")
TOOL_CHOICES = ["initial", "created", "manual"]
FALLBACK_CLIENTS = ["Claude_Haiku_4_5", "Claude_Sonnet_4_6"]
# How long the server keeps running after the last browser tab disconnects,
# before the tab-close killswitch stops it. Wide enough to absorb a page refresh
# (a brief disconnect + reconnect) without a false kill.
TAB_CLOSE_GRACE_SECONDS = 15.0


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
        # Tab-close killswitch: stop the server once no browser tab has been
        # connected for grace_seconds. Read live by the watchdog, so toggling
        # takes effect even while the server is running.
        self.kill_on_tab_close: bool = True
        self.grace_seconds: float = TAB_CLOSE_GRACE_SECONDS
        self._killswitch_warned: bool = False

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
            self._killswitch_warned = False
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
            threading.Thread(
                target=self._kill_watch, args=(self.proc,), daemon=True, name="tab-killswitch"
            ).start()

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

    def _kill_watch(self, proc: subprocess.Popen[str]) -> None:
        """Tab-close killswitch. Stops the server once no browser tab has been
        connected for grace_seconds, while kill_on_tab_close is set. Tied to one
        spawned `proc`: returns as soon as that process is replaced or gone, so a
        fresh start() always owns its own watchdog."""
        zero_since: float | None = None
        while True:
            time.sleep(1.0)
            if self.proc is not proc or proc.poll() is not None:
                return  # this server was stopped or replaced by a new start()
            if not self.kill_on_tab_close:
                zero_since = None
                continue
            n = active_session_count()
            if n is None:
                if not self._killswitch_warned:
                    self._killswitch_warned = True
                    self.log.append(
                        "\n[control] killswitch off: can't read Streamlit session count.\n"
                    )
                continue
            if n > 0:
                zero_since = None
                continue
            # No tab connected. Start (or keep) the grace countdown.
            if zero_since is None:
                zero_since = time.monotonic()
            elif time.monotonic() - zero_since >= self.grace_seconds:
                self.log.append("\n[control] tab closed — stopping server (killswitch).\n")
                self.stop()
                return


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


def active_session_count() -> int | None:
    """Number of *connected* Streamlit sessions (browser tabs), read from the
    runtime. A closed tab drops out immediately; a refresh reconnects within a
    second or two; a backgrounded tab stays connected. Returns None if the count
    can't be read — in which case the killswitch holds off rather than guessing."""
    try:
        from streamlit.runtime import get_instance

        return get_instance()._session_mgr.num_active_sessions()
    except Exception:  # noqa: BLE001 - any failure just disables the killswitch
        return None


def arm_beforeunload(armed: bool) -> None:
    """(Un)install a `beforeunload` confirm dialog on the page so the tab can't
    be closed by accident while doing so would stop the server. Always renders
    (idempotently add/remove) so disarming reliably tears the listener down — a
    one-shot conditional render would orphan it on the parent window."""
    flag = "true" if armed else "false"
    components.html(
        f"""
        <script>
        (function() {{
          var ARMED = {flag};
          function arm(w) {{
            try {{
              if (w.__cobbieBUL) {{
                w.removeEventListener('beforeunload', w.__cobbieBUL);
                w.__cobbieBUL = null;
              }}
              if (ARMED) {{
                var h = function(e) {{ e.preventDefault(); e.returnValue = ''; return ''; }};
                w.__cobbieBUL = h;
                w.addEventListener('beforeunload', h);
              }}
            }} catch (e) {{}}
          }}
          arm(window);
          try {{ if (window.parent && window.parent !== window) arm(window.parent); }} catch (e) {{}}
        }})();
        </script>
        """,
        height=0,
    )


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

    st.divider()
    kill_on_close = st.toggle(
        "Stop server when I close this tab", value=True,
        help=(
            f"On: closing this tab stops the server ~{int(TAB_CLOSE_GRACE_SECONDS)}s later "
            "(a refresh is safe; you'll get a confirm prompt first). Off: the server keeps "
            "running after the tab closes — keep the headset served while you step away."
        ),
    )

# Apply the killswitch settings live, so toggling takes effect even while running.
sup.kill_on_tab_close = kill_on_close
sup.grace_seconds = TAB_CLOSE_GRACE_SECONDS

st.title("Cobbie Server Control")
st.caption(
    "Configure and run the Cobbie WebSocket server. By default, closing this tab stops the server "
    "(after a short grace window, with a confirm prompt) — untick the sidebar toggle to leave it "
    "running for the headset. You can also stop it explicitly here, or quit this Streamlit process."
)

running = sup.is_running()
# Guard the tab against accidental close only while doing so would actually stop
# the server (server up + killswitch armed). Renders every run so Stop/untoggle
# tears the prompt back down.
arm_beforeunload(running and kill_on_close)
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
