"""Auto-installation logic for ComfyUI and HY-Motion plugin."""

import subprocess
import sys
import shutil
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import (
    COMFYUI_DIR,
    MODELS_DIR,
    BASE_DIR,
    MODELS,
    ensure_base_directory,
    ensure_model_directory,
    is_installed,
    mark_installed,
)

console = Console()


def run_command(
    cmd: list[str],
    description: str,
    cwd: Path | None = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    """Run a command with progress indication."""
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
        transient=True,
    ) as progress:
        progress.add_task(description=description, total=None)
        result = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
        )
        if check and result.returncode != 0:
            console.print(f"[red]Error:[/red] {result.stderr or result.stdout}")
            raise RuntimeError(f"Command failed: {' '.join(cmd)}")
        return result


def ensure_comfy_cli():
    """Ensure comfy-cli is installed."""
    if shutil.which("comfy"):
        return

    console.print("[yellow]Installing comfy-cli...[/yellow]")
    run_command(
        [sys.executable, "-m", "pip", "install", "comfy-cli"],
        "Installing comfy-cli",
    )


def is_comfyui_actually_installed() -> bool:
    """Check if ComfyUI is actually installed (not just directory exists)."""
    main_py = COMFYUI_DIR / "main.py"
    return main_py.exists()


def install_comfyui():
    """Install ComfyUI to the workspace directory."""
    console.print(f"[yellow]Installing ComfyUI to {COMFYUI_DIR}...[/yellow]")

    # Use comfy-cli to install with --skip-prompt for non-interactive mode
    # Include ComfyUI-Manager since we need it for node installation
    run_command(
        [
            "comfy",
            "--skip-prompt",
            f"--workspace={COMFYUI_DIR}",
            "install",
            "--nvidia",  # Assume NVIDIA GPU
        ],
        "Installing ComfyUI (this may take a few minutes)",
    )


def install_hy_motion_plugin():
    """Install the ComfyUI-HY-Motion1 plugin."""
    console.print("[yellow]Installing HY-Motion plugin...[/yellow]")

    # Install the plugin using comfy-cli with GitHub URL
    run_command(
        [
            "comfy",
            "--skip-prompt",
            f"--workspace={COMFYUI_DIR}",
            "node",
            "install",
            "https://github.com/jtydhr88/ComfyUI-HY-Motion1",
        ],
        "Installing HY-Motion plugin",
    )


def install_fbxsdkpy():
    """Install fbxsdkpy for FBX export with Mixamo retargeting."""
    console.print("[yellow]Installing fbxsdkpy for FBX export...[/yellow]")

    # fbxsdkpy requires the PyNimation package registry
    # This provides pre-built wheels for Python 3.10-3.13
    try:
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "fbxsdkpy",
                "--extra-index-url",
                "https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple",
            ],
            "Installing fbxsdkpy",
            check=False,  # Don't fail if this doesn't work
        )
    except Exception:
        console.print(
            "[yellow]Warning:[/yellow] fbxsdkpy installation failed. "
            "FBX export requires the Autodesk FBX SDK. "
            "For full FBX support, install fbxsdkpy manually."
        )


def download_models(model_key: str):
    """Download HY-Motion model weights from HuggingFace."""
    from huggingface_hub import snapshot_download

    model_config = MODELS[model_key]
    model_name = model_config["name"]

    console.print(f"[yellow]Downloading {model_name} model weights...[/yellow]")

    # Create the model directory
    model_dir = MODELS_DIR / "ckpts" / "tencent" / model_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Use huggingface_hub Python API (avoids Windows encoding issues with CLI)
    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
            transient=True,
        ) as progress:
            progress.add_task(
                description=f"Downloading {model_name} (this may take several minutes)",
                total=None
            )
            snapshot_download(
                repo_id=model_config["repo"],
                local_dir=str(MODELS_DIR / "ckpts" / "tencent"),
                allow_patterns=[f"{model_name}/*"],
            )
        console.print(f"[green]Downloaded {model_name} successfully.[/green]")
    except Exception as e:
        console.print(f"[yellow]Warning:[/yellow] Model download failed: {e}")
        console.print("Models will be downloaded automatically on first use.")


def download_text_encoder():
    """Download the text encoder model (Qwen3-8B or CLIP)."""
    console.print("[yellow]Text encoder will be downloaded on first use.[/yellow]")
    # The ComfyUI plugin handles this automatically


def ensure_comfyui_installed(model_key: str = "lite", force: bool = False):
    """Ensure ComfyUI and all required components are installed.

    Args:
        model_key: Which model to download ("full" or "lite")
        force: Force reinstallation even if already installed
    """
    # Only create base directory before ComfyUI install
    ensure_base_directory()

    if is_installed() and not force:
        console.print("[green]ComfyUI is already installed.[/green]")
        return

    console.print("[bold blue]Setting up HY-Motion Exporter...[/bold blue]")
    console.print(f"Installation directory: {BASE_DIR}")
    console.print()

    # Step 1: Install comfy-cli
    ensure_comfy_cli()

    # Step 2: Install ComfyUI (comfy-cli will create the directory)
    if not is_comfyui_actually_installed() or force:
        install_comfyui()

    # Now that ComfyUI is installed, create model directory
    ensure_model_directory()

    # Step 3: Install HY-Motion plugin
    install_hy_motion_plugin()

    # Step 4: Install fbxsdkpy
    install_fbxsdkpy()

    # Step 5: Download models
    download_models(model_key)

    # Step 6: Note about text encoder
    download_text_encoder()

    # Mark installation complete
    mark_installed()

    console.print()
    console.print("[bold green]Installation complete![/bold green]")


def check_installation_status() -> dict:
    """Check the status of all installed components."""
    status = {
        "base_dir_exists": BASE_DIR.exists(),
        "comfyui_installed": COMFYUI_DIR.exists(),
        "install_marker": is_installed(),
        "comfy_cli_available": shutil.which("comfy") is not None,
    }

    # Check for model files
    for model_key, model_config in MODELS.items():
        model_dir = MODELS_DIR / "ckpts" / "tencent" / model_config["name"]
        status[f"model_{model_key}_downloaded"] = model_dir.exists()

    return status
