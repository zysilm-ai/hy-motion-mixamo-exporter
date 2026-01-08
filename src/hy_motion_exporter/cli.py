"""CLI entry point for HY-Motion Mixamo Exporter."""

import shutil
import sys
from pathlib import Path

import click
from rich.console import Console

from . import __version__
from .config import select_model, select_llm_precision, get_vram_gb, DEFAULT_PORT, is_installed, LLM_PRECISIONS
from .setup import ensure_comfyui_installed
from .server import get_server
from .workflow import generate_workflow, execute_workflow

console = Console(force_terminal=False, legacy_windows=True)


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
    required=True,
    help="Mixamo character FBX file (required). Download from mixamo.com",
)
@click.option(
    "--duration", "-d",
    type=float,
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
    "--port", "-p",
    type=int,
    default=DEFAULT_PORT,
    help="ComfyUI server port",
)
@click.option(
    "--keep-server",
    is_flag=True,
    help="Keep ComfyUI server running after generation",
)
@click.option(
    "--reinstall",
    is_flag=True,
    help="Force reinstall ComfyUI and plugins",
)
@click.version_option(version=__version__)
def main(
    prompt: str,
    output: str,
    character: str | None,
    duration: float | None,
    model: str,
    precision: str,
    seed: int | None,
    port: int,
    keep_server: bool,
    reinstall: bool,
):
    """Generate Mixamo-compatible motion from a text prompt.

    PROMPT is the text description of the motion you want to generate.

    Examples:

    \b
      hy-motion-export "A person walks forward" -c character.fbx
      hy-motion-export "Dancing happily" -c character.fbx -o dance.fbx
      hy-motion-export "Running" -c character.fbx -d 5.0 --keep-server
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

    # Ensure ComfyUI is installed
    if reinstall or not is_installed():
        try:
            ensure_comfyui_installed(model_key=selected_model, force=reinstall)
        except Exception as e:
            console.print(f"[red]Installation failed: {e}[/red]")
            sys.exit(1)

    # Start the server
    server = get_server(port)
    if not server.start():
        console.print("[red]Failed to start ComfyUI server.[/red]")
        sys.exit(1)

    try:
        # Generate workflow
        console.print()
        console.print(f'[bold]Prompt:[/bold] "{prompt}"')
        if duration:
            console.print(f"[dim]Duration: {duration}s[/dim]")
        if character:
            console.print(f"[dim]Character: {character}[/dim]")
        console.print()

        workflow = generate_workflow(
            prompt=prompt,
            duration=duration,
            character_fbx=character,
            model=selected_model,
            llm_precision=selected_precision,
            seed=seed,
            filename_prefix=output_path.stem,
        )

        # Execute workflow
        result_path = execute_workflow(
            workflow=workflow,
            server_url=server.url,
            output_dir=output_path.parent,
        )

        if result_path:
            # Rename to desired output name if different
            if result_path.name != output_path.name:
                final_path = output_path.parent / output_path.name
                if final_path.exists():
                    final_path.unlink()
                shutil.move(result_path, final_path)
                result_path = final_path

            console.print()
            console.print(f"[bold green]Success![/bold green] Output saved to: {result_path}")
        else:
            console.print("[red]Failed to generate motion.[/red]")
            sys.exit(1)

    finally:
        # Stop server unless keep-server is set
        if not keep_server:
            server.stop()


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
    "--port", "-p",
    type=int,
    default=DEFAULT_PORT,
    help="ComfyUI server port",
)
def stop(port: int):
    """Stop the ComfyUI server and free VRAM."""
    server = get_server(port)
    server.stop()


@click.command()
@click.option(
    "--yes", "-y",
    is_flag=True,
    help="Skip confirmation prompt",
)
def uninstall(yes: bool):
    """Uninstall ComfyUI and all downloaded models."""
    from .config import BASE_DIR, INSTALL_MARKER
    import shutil

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

    # Stop server first
    console.print("[yellow]Stopping server...[/yellow]")
    server = get_server(DEFAULT_PORT)
    server.stop()

    # Delete the directory
    console.print("[yellow]Removing files...[/yellow]")
    try:
        shutil.rmtree(BASE_DIR)
        console.print("[green]Uninstall complete. All files removed.[/green]")
    except Exception as e:
        console.print(f"[red]Failed to remove some files: {e}[/red]")
        console.print(f"[yellow]Please manually delete: {BASE_DIR}[/yellow]")


@click.group()
def cli():
    """HY-Motion Mixamo Exporter - Generate motion from text."""
    pass


cli.add_command(main, name="generate")
cli.add_command(status)
cli.add_command(stop)


# Allow running as `hy-motion-export "prompt"` directly
if __name__ == "__main__":
    main()
