"""Orchestrate llama-server processes — start, health-check, stop."""

import logging
import os
import signal
import subprocess
import time
from typing import Optional

import requests

from .config import ModelConfig

log = logging.getLogger(__name__)


def _build_server_cmd(config: ModelConfig) -> list[str]:
    """Build the llama-server command line from a ModelConfig."""
    cmd: list[str] = []

    # Optional taskset prefix
    if config.taskset_cpus:
        cmd.extend(["taskset", "-c", config.taskset_cpus])

    cmd.append(config.server_binary)
    cmd.extend(["-m", config.model_path])
    cmd.extend(["--alias", config.alias])
    cmd.extend(["--host", config.host])
    cmd.extend(["--port", str(config.port)])
    cmd.extend(["--api-key", config.api_key])

    # Extra flags
    for key, value in config.server_args.items():
        flag = f"--{key}" if len(key) > 1 else f"-{key}"
        if isinstance(value, bool):
            if value:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(value)])

    return cmd


def is_server_ready(port: int, host: str = "127.0.0.1") -> bool:
    """Check if the llama-server /health endpoint returns OK."""
    try:
        r = requests.get(f"http://{host}:{port}/health", timeout=3)
        return r.status_code == 200
    except (requests.ConnectionError, requests.Timeout):
        return False


def wait_for_server(
    port: int,
    host: str = "127.0.0.1",
    timeout: float = 300,
    poll_interval: float = 3.0,
) -> bool:
    """Block until the server's /health endpoint returns 200 or timeout."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if is_server_ready(port, host):
            return True
        time.sleep(poll_interval)
    return False


def start_server(
    config: ModelConfig,
    timeout: float = 300,
    log_file: Optional[str] = None,
) -> subprocess.Popen:
    """Launch llama-server and wait for it to become healthy.

    Returns the Popen object.  Raises RuntimeError if the server
    does not become ready within *timeout* seconds.
    """
    cmd = _build_server_cmd(config)
    env = {**os.environ, **config.env}

    log.info("Starting server: %s", " ".join(cmd))

    log_fh = open(log_file, "w") if log_file else subprocess.DEVNULL
    proc = subprocess.Popen(
        cmd,
        env=env,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        preexec_fn=os.setsid,  # own process group for clean shutdown
    )

    if not wait_for_server(config.port, timeout=timeout):
        stop_server(proc)
        raise RuntimeError(
            f"Server for {config.name} did not become ready within {timeout}s"
        )

    log.info("Server %s is ready on port %d", config.name, config.port)
    return proc


def stop_server(proc: subprocess.Popen, timeout: float = 10) -> None:
    """Gracefully stop a llama-server process."""
    if proc.poll() is not None:
        return  # already exited

    log.info("Stopping server (pid=%d)…", proc.pid)
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=timeout)
    except (subprocess.TimeoutExpired, ProcessLookupError):
        log.warning("Server did not exit gracefully, sending SIGKILL…")
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=5)
        except Exception:
            pass
    log.info("Server stopped.")
