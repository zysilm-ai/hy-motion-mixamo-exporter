"""Workflow generation and ComfyUI API interaction."""

import json
import random
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import requests
import websocket
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from .config import COMFYUI_DIR, MODELS

# Use ASCII-safe console to avoid Windows encoding issues
console = Console(force_terminal=False, legacy_windows=True)

# Path to workflow template
WORKFLOW_TEMPLATE_PATH = Path(__file__).parent / "workflows" / "text_to_mixamo.json"


def load_workflow_template() -> dict:
    """Load the base workflow template."""
    with open(WORKFLOW_TEMPLATE_PATH) as f:
        return json.load(f)


def generate_workflow(
    prompt: str,
    duration: float | None = None,
    character_fbx: str | None = None,
    model: str = "lite",
    llm_precision: str = "int8",
    seed: int | None = None,
    filename_prefix: str = "hy_motion",
) -> dict:
    """Generate a ComfyUI workflow for text-to-Mixamo motion.

    Args:
        prompt: Text description of the motion
        duration: Motion duration in seconds (auto-detected if None)
        character_fbx: Path to custom Mixamo character FBX
        model: Model variant ("full" or "lite")
        llm_precision: LLM quantization ("none", "int8", "int4", or "gguf")
        seed: Random seed for reproducibility
        filename_prefix: Prefix for output files

    Returns:
        Workflow dictionary ready for API submission
    """
    workflow = load_workflow_template()

    # Model name
    model_name = MODELS[model]["name"]

    # Generate seed if not provided
    if seed is None:
        seed = random.randint(0, 2**31 - 1)  # Max INT32

    # Default duration if not specified (node requires 0.5-12.0 seconds)
    if duration is None:
        duration = 3.0  # Default to 3 seconds

    # Character FBX is required for Mixamo retargeting
    if not character_fbx:
        console.print("[red]Character FBX file is required. Download from mixamo.com[/red]")
        return None

    # Copy character FBX to ComfyUI input directory
    character_path = Path(character_fbx)
    if not character_path.exists():
        console.print(f"[red]Character file not found: {character_fbx}[/red]")
        return None
    input_dir = COMFYUI_DIR / "input"
    input_dir.mkdir(parents=True, exist_ok=True)
    dest_path = input_dir / character_path.name
    shutil.copy2(character_path, dest_path)
    custom_fbx_path = character_path.name  # Just filename, plugin looks in input/

    # Modify workflow values directly
    workflow["1"]["inputs"]["quantization"] = llm_precision  # "none", "int8", or "int4"
    workflow["1"]["inputs"]["offload_to_cpu"] = False
    workflow["2"]["inputs"]["model_name"] = model_name
    workflow["3"]["inputs"]["text"] = prompt
    workflow["4"]["inputs"]["duration"] = float(duration)
    workflow["4"]["inputs"]["seed"] = seed
    workflow["5"]["inputs"]["filename_prefix"] = filename_prefix
    workflow["5"]["inputs"]["custom_fbx_path"] = custom_fbx_path.replace("\\", "/") if custom_fbx_path else ""

    return workflow


def execute_workflow(
    workflow: dict,
    server_url: str,
    output_dir: Path | None = None,
) -> Path | None:
    """Execute a workflow on the ComfyUI server and wait for completion.

    Args:
        workflow: The workflow dictionary
        server_url: ComfyUI server URL
        output_dir: Directory to copy output files to

    Returns:
        Path to the generated FBX file, or None if failed
    """
    client_id = str(uuid.uuid4())

    # Submit the workflow
    prompt_data = {
        "prompt": workflow,
        "client_id": client_id,
    }

    console.print("[yellow]Submitting workflow...[/yellow]")

    try:
        response = requests.post(
            f"{server_url}/prompt",
            json=prompt_data,
            timeout=600,  # 10 minutes for first-time model loading
        )
        if response.status_code != 200:
            console.print(f"[red]Server response: {response.text}[/red]")
        response.raise_for_status()
        result = response.json()
        prompt_id = result["prompt_id"]
    except requests.RequestException as e:
        console.print(f"[red]Failed to submit workflow: {e}[/red]")
        return None

    console.print(f"[green]Workflow submitted. Prompt ID: {prompt_id}[/green]")

    # Connect to WebSocket for progress updates
    ws_url = server_url.replace("http://", "ws://").replace("https://", "wss://")
    ws_url = f"{ws_url}/ws?clientId={client_id}"

    output_files: list[str] = []

    try:
        ws = websocket.create_connection(ws_url, timeout=60)
        ws.settimeout(30)  # 30 second timeout for each recv()

        print("Generating motion...", end="", flush=True)
        last_activity = time.time()
        max_idle_time = 600  # 10 minutes max idle time
        last_node = None

        while True:
            try:
                message = ws.recv()
                last_activity = time.time()
                data = json.loads(message)

                if data.get("type") == "executing":
                    node = data["data"].get("node")
                    if node and node != last_node:
                        print(".", end="", flush=True)
                        last_node = node
                    elif data["data"].get("prompt_id") == prompt_id:
                        # Execution complete
                        print(" Done!")
                        break

                elif data.get("type") == "executed":
                    node_output = data["data"].get("output", {})
                    # Check for FBX output files
                    if "fbx_files" in node_output:
                        output_files.extend(node_output["fbx_files"])
                    elif "files" in node_output:
                        for f in node_output["files"]:
                            if f.get("filename", "").endswith(".fbx"):
                                output_files.append(f["filename"])

            except websocket.WebSocketTimeoutException:
                print(".", end="", flush=True)
                # Check if we've been idle too long
                if time.time() - last_activity > max_idle_time:
                    print("\nTimeout waiting for server response.")
                    break

                # Check if execution is complete via REST API
                try:
                    history = requests.get(
                        f"{server_url}/history/{prompt_id}",
                        timeout=10,
                    ).json()
                    if prompt_id in history:
                        print(" Done!")
                        break
                except Exception:
                    pass  # Server might be busy, continue waiting

        ws.close()

    except Exception as e:
        print(f"\nWebSocket monitoring failed: {e}")
        print("Falling back to polling...")

        # Fallback: poll the history endpoint
        output_files = _poll_for_completion(server_url, prompt_id)

    # Find the FBX file in ComfyUI output directory
    comfyui_output = COMFYUI_DIR / "output"
    fbx_path = None

    # First try to find by filename from WebSocket messages
    for filename in output_files:
        potential_path = comfyui_output / filename
        if potential_path.exists():
            fbx_path = potential_path
            break
        # Also check in subdirectories
        for sub_path in comfyui_output.glob(f"**/{filename}"):
            if sub_path.exists():
                fbx_path = sub_path
                break

    # Fall back to searching for recent FBX files
    # Use recursive glob since files may be in subdirectories like hymotion_output/
    if fbx_path is None:
        fbx_files = list(comfyui_output.glob("**/*.fbx"))
        if fbx_files:
            # Get the most recently modified FBX file
            fbx_path = max(fbx_files, key=lambda p: p.stat().st_mtime)

    if fbx_path is None:
        console.print("[red]Could not find generated FBX file.[/red]")
        return None

    # Copy to output directory if specified
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        dest_path = output_dir / fbx_path.name
        shutil.copy2(fbx_path, dest_path)
        return dest_path

    return fbx_path


def _poll_for_completion(
    server_url: str,
    prompt_id: str,
    timeout: int = 600,
    poll_interval: int = 2,
) -> list[str]:
    """Poll the history endpoint until execution completes.

    Args:
        server_url: ComfyUI server URL
        prompt_id: The prompt ID to monitor
        timeout: Maximum seconds to wait
        poll_interval: Seconds between polls

    Returns:
        List of output filenames
    """
    start_time = time.time()
    output_files = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        task = progress.add_task("Waiting for generation to complete...", total=None)

        while time.time() - start_time < timeout:
            try:
                response = requests.get(
                    f"{server_url}/history/{prompt_id}",
                    timeout=10,
                )
                history = response.json()

                if prompt_id in history:
                    prompt_data = history[prompt_id]
                    outputs = prompt_data.get("outputs", {})

                    for node_id, node_output in outputs.items():
                        if "files" in node_output:
                            for f in node_output["files"]:
                                filename = f.get("filename", "")
                                if filename.endswith(".fbx"):
                                    output_files.append(filename)

                    if outputs:  # Execution complete
                        break

            except requests.RequestException:
                pass

            time.sleep(poll_interval)

    return output_files


def get_queue_status(server_url: str) -> dict[str, Any]:
    """Get the current queue status from the server."""
    try:
        response = requests.get(f"{server_url}/queue", timeout=10)
        return response.json()
    except requests.RequestException:
        return {}
