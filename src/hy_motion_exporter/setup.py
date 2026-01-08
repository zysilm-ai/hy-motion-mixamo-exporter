"""Auto-installation logic for HY-Motion models and dependencies."""

import subprocess
import sys
import shutil
from pathlib import Path

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from .config import (
    MODELS_DIR,
    BASE_DIR,
    MODELS,
    ensure_base_directory,
    ensure_model_directory,
    is_installed,
    mark_installed,
)

console = Console(force_terminal=False, legacy_windows=True)


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
        console.print("[green]fbxsdkpy installed successfully.[/green]")
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

    # Check if already downloaded
    model_dir = MODELS_DIR / "ckpts" / "tencent" / model_name
    if model_dir.exists() and (model_dir / "config.yml").exists():
        console.print(f"[green]{model_name} already downloaded.[/green]")
        return

    console.print(f"[yellow]Downloading {model_name} model weights...[/yellow]")

    # Create the model directory
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


def check_pytorch_installed() -> bool:
    """Check if PyTorch is installed with CUDA support."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


def check_transformers_installed() -> bool:
    """Check if transformers is installed."""
    try:
        import transformers
        return True
    except ImportError:
        return False


def install_dependencies():
    """Install required Python dependencies."""
    deps_to_install = []

    # Check PyTorch
    if not check_pytorch_installed():
        console.print("[yellow]PyTorch not found or CUDA not available.[/yellow]")
        console.print("Please install PyTorch with CUDA support from https://pytorch.org/")
        console.print("Example: pip install torch --index-url https://download.pytorch.org/whl/cu124")
        raise RuntimeError("PyTorch with CUDA support is required")

    # Check transformers
    if not check_transformers_installed():
        deps_to_install.append("transformers>=4.40")
        deps_to_install.append("accelerate>=0.30")
        deps_to_install.append("bitsandbytes>=0.43")

    # Install missing dependencies
    if deps_to_install:
        console.print("[yellow]Installing required dependencies...[/yellow]")
        run_command(
            [sys.executable, "-m", "pip", "install"] + deps_to_install,
            "Installing dependencies",
        )


def ensure_installed(model_key: str = "lite", force: bool = False):
    """Ensure all required components are installed.

    Args:
        model_key: Which model to download ("full" or "lite")
        force: Force reinstallation even if already installed
    """
    ensure_base_directory()
    ensure_model_directory()

    if is_installed() and not force:
        console.print("[green]HY-Motion is already set up.[/green]")
        return

    console.print("[bold blue]Setting up HY-Motion Exporter...[/bold blue]")
    console.print(f"Installation directory: {BASE_DIR}")
    console.print()

    # Step 1: Check/install dependencies
    install_dependencies()

    # Step 2: Install fbxsdkpy
    install_fbxsdkpy()

    # Step 3: Download models
    download_models(model_key)

    # Step 4: Note about text encoder
    console.print("[yellow]Text encoder (Qwen3-8B + CLIP) will be downloaded on first use.[/yellow]")

    # Mark installation complete
    mark_installed()

    console.print()
    console.print("[bold green]Setup complete![/bold green]")


def check_installation_status() -> dict:
    """Check the status of all installed components."""
    status = {
        "base_dir_exists": BASE_DIR.exists(),
        "install_marker": is_installed(),
        "pytorch_available": check_pytorch_installed(),
        "transformers_available": check_transformers_installed(),
    }

    # Check for model files
    for model_key, model_config in MODELS.items():
        model_dir = MODELS_DIR / "ckpts" / "tencent" / model_config["name"]
        config_exists = (model_dir / "config.yml").exists()
        ckpt_exists = (model_dir / "latest.ckpt").exists()
        status[f"model_{model_key}_downloaded"] = config_exists and ckpt_exists

    # Check fbxsdkpy
    try:
        import fbx
        status["fbxsdkpy_available"] = True
    except ImportError:
        status["fbxsdkpy_available"] = False

    return status


# Keep old function name for backwards compatibility
ensure_comfyui_installed = ensure_installed
