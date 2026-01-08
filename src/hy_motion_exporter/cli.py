"""CLI entry point for HY-Motion Mixamo Exporter."""

import shutil
import sys
from pathlib import Path

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from . import __version__
from .config import (
    select_model,
    select_llm_precision,
    get_vram_gb,
    is_installed,
    MODELS,
    BASE_DIR,
    get_model_path,
)
from .setup import ensure_installed

console = Console(force_terminal=False, legacy_windows=True)


def progress_callback(step: int, total: int):
    """Progress callback for generation."""
    # This is called during generation steps
    pass


@click.command()
@click.argument("prompt")
@click.option(
    "--output", "-o",
    type=click.Path(),
    default="output.fbx",
    help="Output FBX file path",
)
@click.option(
    "--character", "-c",
    type=click.Path(exists=True),
    help="Mixamo character FBX file for retargeting (optional)",
)
@click.option(
    "--duration", "-d",
    type=float,
    default=3.0,
    help="Motion duration in seconds, 0.5-12.0 (default: 3.0)",
)
@click.option(
    "--model", "-m",
    type=click.Choice(["auto", "full", "lite"]),
    default="auto",
    help="Model variant to use (full=1B params, lite=0.46B params)",
)
@click.option(
    "--precision",
    type=click.Choice(["auto", "none", "int8", "int4"]),
    default="auto",
    help="LLM quantization (none=16GB, int8=8GB, int4=4GB VRAM)",
)
@click.option(
    "--seed", "-s",
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--cfg-scale",
    type=float,
    default=5.0,
    help="Classifier-free guidance scale (default: 5.0)",
)
@click.option(
    "--reinstall",
    is_flag=True,
    help="Force reinstall models",
)
@click.version_option(version=__version__)
def main(
    prompt: str,
    output: str,
    character: str | None,
    duration: float,
    model: str,
    precision: str,
    seed: int | None,
    cfg_scale: float,
    reinstall: bool,
):
    """Generate Mixamo-compatible motion from a text prompt.

    PROMPT is the text description of the motion you want to generate.

    Examples:

    \b
      hy-motion-export "A person walks forward" -o walk.fbx
      hy-motion-export "Dancing happily" -c character.fbx -o dance.fbx
      hy-motion-export "Running" -d 5.0 -o running.fbx
    """
    console.print(f"[bold blue]HY-Motion Mixamo Exporter v{__version__}[/bold blue]")
    console.print()

    # Validate output path
    output_path = Path(output)
    if output_path.suffix.lower() != ".fbx":
        output_path = output_path.with_suffix(".fbx")

    # Select model and precision based on VRAM
    try:
        selected_model = select_model(model)
        selected_precision = select_llm_precision(precision)
        vram = get_vram_gb()
        if vram:
            console.print(f"[dim]GPU VRAM: {vram:.1f}GB[/dim]")
        console.print(f"[dim]Model: {selected_model}, LLM precision: {selected_precision}[/dim]")
    except RuntimeError as e:
        console.print(f"[red]Error: {e}[/red]")
        sys.exit(1)

    # Ensure models are installed
    if reinstall or not is_installed():
        try:
            ensure_installed(model_key=selected_model, force=reinstall)
        except Exception as e:
            console.print(f"[red]Setup failed: {e}[/red]")
            sys.exit(1)

    # Print generation info
    console.print()
    console.print(f'[bold]Prompt:[/bold] "{prompt}"')
    console.print(f"[dim]Duration: {duration}s[/dim]")
    if character:
        console.print(f"[dim]Character: {character}[/dim]")
    console.print()

    try:
        # Import inference module (delayed to avoid slow imports on help)
        from .inference import HYMotionInference
        from .export import export_fbx, check_fbx_available

        # Check FBX availability
        if not check_fbx_available():
            console.print("[yellow]Warning: fbxsdkpy not installed. FBX export may fail.[/yellow]")
            console.print("[yellow]Install with: pip install fbxsdkpy --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple[/yellow]")
            console.print()

        # Initialize inference
        model_path = get_model_path(selected_model)
        model_name = MODELS[selected_model]["name"]

        console.print("[yellow]Loading model...[/yellow]")
        inference = HYMotionInference(
            model_name=model_name,
            quantization=selected_precision,
            device="cuda",
        )

        # Generate motion with progress
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
            transient=True,
        ) as progress:
            task = progress.add_task("Generating motion...", total=50)

            def update_progress(step, total):
                progress.update(task, completed=step, total=total)

            motion_data = inference.generate(
                prompt=prompt,
                duration=duration,
                seed=seed,
                cfg_scale=cfg_scale,
                progress_callback=update_progress,
            )

        # Export to FBX
        console.print("[yellow]Exporting FBX...[/yellow]")
        result_path = export_fbx(
            motion_data=motion_data,
            output_path=str(output_path),
            character_fbx=character,
        )

        console.print()
        console.print(f"[bold green]Success![/bold green] Output saved to: {result_path}")

        # Cleanup
        inference.unload()

    except ImportError as e:
        console.print(f"[red]Import error: {e}[/red]")
        console.print("[yellow]Please ensure PyTorch and transformers are installed.[/yellow]")
        sys.exit(1)
    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        sys.exit(1)


@click.command()
def status():
    """Check installation status."""
    from .setup import check_installation_status

    console.print("[bold]Installation Status[/bold]")
    console.print()

    status = check_installation_status()
    for key, value in status.items():
        icon = "[green]OK[/green]" if value else "[red]Missing[/red]"
        console.print(f"  {key}: {icon}")


@click.command()
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompt",
)
def uninstall(yes: bool):
    """Uninstall HY-Motion models and cached data."""
    if not BASE_DIR.exists():
        console.print("[yellow]Nothing to uninstall. HY-Motion Exporter is not installed.[/yellow]")
        return

    # Show what will be deleted
    console.print(f"[bold]This will delete:[/bold]")
    console.print(f"  {BASE_DIR}")
    console.print()

    # Calculate size
    total_size = sum(f.stat().st_size for f in BASE_DIR.rglob('*') if f.is_file())
    size_gb = total_size / (1024 ** 3)
    console.print(f"[dim]Total size: {size_gb:.2f} GB[/dim]")
    console.print()

    if not yes:
        confirm = click.confirm("Are you sure you want to uninstall?", default=False)
        if not confirm:
            console.print("[yellow]Uninstall cancelled.[/yellow]")
            return

    # Delete the directory
    console.print("[yellow]Removing files...[/yellow]")
    try:
        shutil.rmtree(BASE_DIR)
        console.print("[green]Uninstall complete. All files removed.[/green]")
    except Exception as e:
        console.print(f"[red]Failed to remove some files: {e}[/red]")
        console.print(f"[yellow]Please manually delete: {BASE_DIR}[/yellow]")


# Allow running as `hy-motion-export "prompt"` directly
if __name__ == "__main__":
    main()
