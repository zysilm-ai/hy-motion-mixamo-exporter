"""
FBX export module with Mixamo retargeting support.

This module handles exporting generated motion to FBX format,
with optional retargeting to Mixamo character skeletons.
"""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

# Check for FBX SDK availability
try:
    import fbx
    FBX_AVAILABLE = True
except ImportError:
    FBX_AVAILABLE = False


def get_assets_dir() -> Path:
    """Get the assets directory path."""
    return Path(__file__).parent / "assets"


def export_fbx(
    motion_data: Dict[str, Any],
    output_path: str,
    character_fbx: Optional[str] = None,
    yaw_offset: float = 0.0,
    scale: float = 0.0,
) -> str:
    """
    Export motion data to FBX format.

    Args:
        motion_data: Motion data from HYMotionInference.generate()
        output_path: Path for the output FBX file
        character_fbx: Optional path to Mixamo character FBX for retargeting
        yaw_offset: Rotation offset in degrees
        scale: Scale factor (0 = auto-detect)

    Returns:
        Path to the exported FBX file

    Raises:
        ImportError: If fbxsdkpy is not installed
        FileNotFoundError: If character_fbx is specified but not found
    """
    if not FBX_AVAILABLE:
        raise ImportError(
            "fbxsdkpy is required for FBX export. Install it with:\n"
            "pip install fbxsdkpy --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple"
        )

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract motion data
    smpl_data = _extract_smpl_data(motion_data)

    if character_fbx:
        # Retarget to Mixamo character
        if not Path(character_fbx).exists():
            raise FileNotFoundError(f"Character FBX not found: {character_fbx}")

        print(f"[Export] Retargeting to Mixamo character: {character_fbx}")
        return _export_with_retargeting(
            smpl_data, output_path, character_fbx, yaw_offset, scale
        )
    else:
        # Export with default wooden boy skeleton
        print("[Export] Exporting with default skeleton")
        return _export_wooden_boy(smpl_data, output_path)


def _extract_smpl_data(motion_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract SMPL data from motion output."""
    # Check if smpl_data is already provided
    if "smpl_data" in motion_data:
        smpl_data_list = motion_data["smpl_data"]
        if isinstance(smpl_data_list, list) and len(smpl_data_list) > 0:
            return smpl_data_list[0]
        return smpl_data_list

    # Otherwise, construct from rot6d and transl
    from .hymotion.pipeline.body_model import construct_smpl_data_dict

    rot6d = motion_data.get("rot6d")
    transl = motion_data.get("transl")

    if rot6d is None or transl is None:
        raise ValueError("Motion data must contain 'rot6d' and 'transl' or 'smpl_data'")

    # Handle batch dimension
    if rot6d.dim() == 4:  # (B, N, J, 6)
        rot6d = rot6d[0]  # Take first sample
        transl = transl[0]

    return construct_smpl_data_dict(rot6d, transl)


def _export_wooden_boy(
    smpl_data: Dict[str, np.ndarray],
    output_path: Path,
) -> str:
    """Export motion using the wooden boy FBX template."""
    from .hymotion.utils.smplh2woodfbx import SMPLH2WoodFBX

    # Initialize converter
    converter = SMPLH2WoodFBX()

    # Convert and save
    success = converter.convert_npz_to_fbx(smpl_data, str(output_path))

    if success:
        print(f"[Export] FBX saved: {output_path}")
        return str(output_path)
    else:
        raise RuntimeError("FBX export failed")


def _export_with_retargeting(
    smpl_data: Dict[str, np.ndarray],
    output_path: Path,
    character_fbx: str,
    yaw_offset: float,
    scale: float,
) -> str:
    """Export motion with Mixamo retargeting."""
    from .hymotion.utils.retarget_fbx import retarget_fbx

    # Create temp file for NPZ data
    with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
        np.savez(tmp.name, **smpl_data)
        npz_path = tmp.name

    try:
        # Perform retargeting
        retarget_fbx(
            npz_path=npz_path,
            target_fbx_path=character_fbx,
            output_path=str(output_path),
            yaw_offset=yaw_offset,
            force_scale=scale if scale > 0 else None,
        )

        print(f"[Export] Retargeted FBX saved: {output_path}")
        return str(output_path)

    finally:
        # Clean up temp file
        if os.path.exists(npz_path):
            os.unlink(npz_path)


def export_npz(
    motion_data: Dict[str, Any],
    output_path: str,
) -> str:
    """
    Export motion data to NPZ format (raw motion data).

    Args:
        motion_data: Motion data from HYMotionInference.generate()
        output_path: Path for the output NPZ file

    Returns:
        Path to the exported NPZ file
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    smpl_data = _extract_smpl_data(motion_data)
    np.savez(str(output_path), **smpl_data)

    print(f"[Export] NPZ saved: {output_path}")
    return str(output_path)


def check_fbx_available() -> bool:
    """Check if FBX SDK is available."""
    return FBX_AVAILABLE
