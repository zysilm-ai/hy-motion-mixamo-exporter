"""ComfyUI server management."""

import atexit
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import psutil
import requests
from rich.console import Console

from .config import (
    COMFYUI_DIR,
    PID_FILE,
    DEFAULT_PORT,
    SERVER_HOST,
    get_server_url,
)

console = Console()


class ComfyUIServer:
    """Manages the ComfyUI server process."""

    def __init__(self, port: int = DEFAULT_PORT):
        self.port = port
        self.process: subprocess.Popen | None = None
        self._registered_cleanup = False

    @property
    def url(self) -> str:
        """Get the server URL."""
        return get_server_url(self.port)

    def is_running(self) -> bool:
        """Check if the server is running and responding."""
        try:
            response = requests.get(f"{self.url}/system_stats", timeout=2)
            return response.status_code == 200
        except requests.RequestException:
            return False

    def _is_port_in_use(self) -> bool:
        """Check if the port is already in use."""
        for conn in psutil.net_connections(kind="inet"):
            if conn.laddr.port == self.port and conn.status == "LISTEN":
                return True
        return False

    def _read_pid(self) -> int | None:
        """Read the PID from the PID file."""
        if PID_FILE.exists():
            try:
                return int(PID_FILE.read_text().strip())
            except (ValueError, OSError):
                pass
        return None

    def _write_pid(self, pid: int):
        """Write the PID to the PID file."""
        PID_FILE.write_text(str(pid))

    def _remove_pid_file(self):
        """Remove the PID file."""
        if PID_FILE.exists():
            PID_FILE.unlink()

    def _wait_for_ready(self, timeout: int = 120) -> bool:
        """Wait for the server to become ready.

        Args:
            timeout: Maximum seconds to wait

        Returns:
            True if server is ready, False if timeout
        """
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.is_running():
                return True
            time.sleep(1)
        return False

    def start(self, wait: bool = True) -> bool:
        """Start the ComfyUI server.

        Args:
            wait: Whether to wait for the server to be ready

        Returns:
            True if server started successfully
        """
        # Check if already running
        if self.is_running():
            console.print("[green]ComfyUI server is already running.[/green]")
            return True

        # Check if port is in use by another process
        if self._is_port_in_use():
            console.print(
                f"[yellow]Port {self.port} is in use. "
                f"Checking if it's a ComfyUI server...[/yellow]"
            )
            if self.is_running():
                console.print("[green]Existing ComfyUI server detected.[/green]")
                return True
            else:
                console.print(
                    f"[red]Port {self.port} is in use by another application.[/red]"
                )
                return False

        console.print(f"[yellow]Starting ComfyUI server on port {self.port}...[/yellow]")

        # Build the command
        # Use comfy-cli to launch
        cmd = [
            "comfy",
            f"--workspace={COMFYUI_DIR}",
            "launch",
            "--",
            "--listen",
            SERVER_HOST,
            "--port",
            str(self.port),
        ]

        # Start the process
        try:
            # On Windows, we need different flags
            if sys.platform == "win32":
                # Use CREATE_NEW_PROCESS_GROUP for Windows
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                )
            else:
                self.process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    start_new_session=True,
                )

            self._write_pid(self.process.pid)

            # Register cleanup handler
            if not self._registered_cleanup:
                atexit.register(self._cleanup)
                self._registered_cleanup = True

        except Exception as e:
            console.print(f"[red]Failed to start ComfyUI server: {e}[/red]")
            return False

        if wait:
            console.print("[yellow]Waiting for server to be ready...[/yellow]")
            if self._wait_for_ready():
                console.print(f"[green]ComfyUI server ready at {self.url}[/green]")
                return True
            else:
                console.print("[red]Server failed to start within timeout.[/red]")
                self.stop()
                return False

        return True

    def stop(self):
        """Stop the ComfyUI server."""
        stopped = False

        # First try to stop our own process
        if self.process is not None:
            try:
                if sys.platform == "win32":
                    self.process.terminate()
                else:
                    os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
                stopped = True
            except Exception:
                # Force kill if graceful shutdown fails
                try:
                    self.process.kill()
                    stopped = True
                except Exception:
                    pass
            self.process = None

        # Also check for orphaned process from PID file
        pid = self._read_pid()
        if pid is not None:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                proc.wait(timeout=10)
                stopped = True
            except (psutil.NoSuchProcess, psutil.TimeoutExpired):
                pass

        # Find and kill process by port if still running
        if not stopped or self._is_port_in_use():
            for conn in psutil.net_connections(kind="inet"):
                if conn.laddr.port == self.port and conn.status == "LISTEN":
                    try:
                        proc = psutil.Process(conn.pid)
                        proc.terminate()
                        proc.wait(timeout=10)
                        stopped = True
                    except (psutil.NoSuchProcess, psutil.TimeoutExpired, psutil.AccessDenied):
                        pass
                    break

        self._remove_pid_file()
        if stopped:
            console.print("[green]ComfyUI server stopped. VRAM freed.[/green]")
        else:
            console.print("[yellow]No ComfyUI server was running.[/yellow]")

    def _cleanup(self):
        """Cleanup handler called on exit."""
        if self.process is not None:
            self.stop()


# Global server instance
_server: ComfyUIServer | None = None


def get_server(port: int = DEFAULT_PORT) -> ComfyUIServer:
    """Get or create the global server instance."""
    global _server
    if _server is None or _server.port != port:
        _server = ComfyUIServer(port)
    return _server
