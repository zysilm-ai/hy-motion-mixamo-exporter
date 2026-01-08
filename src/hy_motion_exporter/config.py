"""Configuration and path management for HY-Motion Exporter."""

import os
import subprocess
import shutil
from pathlib import Path
from typing import Literal

# Base directory for all exporter data
BASE_DIR = Path.home() / ".hy-motion-exporter"
MODELS_DIR = BASE_DIR / "models" / "HY-Motion"
INSTALL_MARKER = BASE_DIR / ".installed"

# Model configurations
# VRAM requirements depend on LLM precision:
#   - Motion model: Full ~8GB, Lite ~4GB
#   - LLM: none ~16GB, int8 ~8GB, int4 ~4GB
# Minimum = motion model + int4 LLM
MODELS = {
    "full": {
        "name": "HY-Motion-1.0",
        "repo": "tencent/HY-Motion-1.0",
        "min_vram_gb": 12,  # 8GB model + 4GB int4 LLM
    },
    "lite": {
        "name": "HY-Motion-1.0-Lite",
        "repo": "tencent/HY-Motion-1.0",
        "min_vram_gb": 8,  # 4GB model + 4GB int4 LLM
    },
}

# LLM (Qwen3-8B) quantization options for text encoding
# This significantly affects VRAM usage
LLM_PRECISIONS = {
    "none": {
        "description": "Full precision",
        "vram_gb": 16,
    },
    "int8": {
        "description": "8-bit quantization",
        "vram_gb": 8,
    },
    "int4": {
        "description": "4-bit quantization",
        "vram_gb": 4,
    },
}


def get_vram_gb() -> float | None:
    """Detect available GPU VRAM in gigabytes.

    Returns None if no GPU is available or detection fails.
    """
    # Try nvidia-smi first (works without torch installed)
    nvidia_smi = shutil.which("nvidia-smi")
    if nvidia_smi:
        try:
            result = subprocess.run(
                [nvidia_smi, "--query-gpu=memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            if result.returncode == 0:
                # Get the first GPU's VRAM (in MB)
                vram_mb = int(result.stdout.strip().split("\n")[0])
                return vram_mb / 1024
        except (subprocess.TimeoutExpired, ValueError, IndexError):
            pass

    # Fallback: try torch if available
    try:
        import torch
        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            return props.total_memory / (1024 ** 3)
    except ImportError:
        pass

    return None


def select_model(preference: Literal["auto", "full", "lite"]) -> str:
    """Select the appropriate model based on preference and available VRAM.

    Args:
        preference: User's model preference ("auto", "full", or "lite")

    Returns:
        Model key ("full" or "lite")

    Raises:
        RuntimeError: If no suitable model can be selected
    """
    if preference in ("full", "lite"):
        return preference

    # Auto-detect based on VRAM
    vram = get_vram_gb()

    if vram is None:
        # Can't detect VRAM, default to lite for safety
        return "lite"

    if vram >= MODELS["full"]["min_vram_gb"]:
        return "full"
    elif vram >= MODELS["lite"]["min_vram_gb"]:
        return "lite"
    else:
        raise RuntimeError(
            f"Insufficient GPU VRAM: {vram:.1f}GB detected, "
            f"minimum {MODELS['lite']['min_vram_gb']}GB required for Lite model. "
            "Please use a GPU with more VRAM."
        )


def select_llm_precision(preference: Literal["auto", "none", "int8", "int4"]) -> str:
    """Select the appropriate LLM quantization based on preference and available VRAM.

    Args:
        preference: User's precision preference

    Returns:
        Quantization key ("none", "int8", or "int4")
    """
    if preference != "auto":
        return preference

    # Auto-detect based on VRAM
    vram = get_vram_gb()

    if vram is None:
        # Can't detect VRAM, default to int4 for safety
        return "int4"

    # Select quantization based on available VRAM
    # Leave some headroom for the motion model
    if vram >= 24:
        return "none"  # Full precision if plenty of VRAM
    elif vram >= 16:
        return "int8"
    else:
        return "int4"  # Most memory efficient


def ensure_base_directory():
    """Create base directory if it doesn't exist."""
    BASE_DIR.mkdir(parents=True, exist_ok=True)


def ensure_model_directory():
    """Create models directory if it doesn't exist."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)


def is_installed() -> bool:
    """Check if required components are installed."""
    return INSTALL_MARKER.exists() and MODELS_DIR.exists()


def mark_installed():
    """Mark the installation as complete."""
    INSTALL_MARKER.touch()


def get_model_path(model_key: str) -> Path:
    """Get the path to a model's checkpoint directory."""
    model_name = MODELS[model_key]["name"]
    return MODELS_DIR / "ckpts" / "tencent" / model_name
