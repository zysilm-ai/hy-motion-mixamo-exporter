"""
FBX export module with Mixamo retargeting.

This module handles exporting generated motion to FBX format,
retargeting to Mixamo character skeletons.
"""

import os
import tempfile
from pathlib import Path
from typing import Any, Dict

import numpy as np

# Check for FBX SDK availability
try:
    import fbx
    FBX_AVAILABLE = True
except ImportError:
    FBX_AVAILABLE = False


def export_fbx(
    motion_data: Dict[str, Any],
    output_path: str,
    character_fbx: str,
    yaw_offset: float = 0.0,
    scale: float = 0.0,
) -> str:
    """
    Export motion data to FBX format with Mixamo retargeting.

    Args:
        motion_data: Motion data from HYMotionInference.generate()
        output_path: Path for the output FBX file
        character_fbx: Path to Mixamo character FBX for retargeting
        yaw_offset: Rotation offset in degrees
        scale: Scale factor (0 = auto-detect)

    Returns:
        Path to the exported FBX file

    Raises:
        ImportError: If fbxsdkpy is not installed
        FileNotFoundError: If character_fbx not found
    """
    if not FBX_AVAILABLE:
        raise ImportError(
            "fbxsdkpy is required for FBX export. Install it with:\n"
            "pip install fbxsdkpy --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple"
        )

    if not Path(character_fbx).exists():
        raise FileNotFoundError(f"Character FBX not found: {character_fbx}")

    # Ensure output directory exists
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract motion data in retarget format
    retarget_data = _extract_retarget_data(motion_data)

    print(f"[Export] Retargeting to Mixamo character: {character_fbx}")
    return _export_with_retargeting(
        retarget_data, output_path, character_fbx, yaw_offset, scale
    )


def _extract_retarget_data(motion_data: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Extract motion data in format expected by retarget_fbx.

    The motion pipeline returns:
    - keypoints3d: (B, T, 52, 3) joint positions
    - rot6d: (B, T, 22, 6) rotation data
    - transl: (B, T, 3) translation
    - root_rotations_mat: (B, T, 3, 3) root rotation matrices
    """
    import torch

    # Get data from motion output
    keypoints3d = motion_data.get("keypoints3d")
    rot6d = motion_data.get("rot6d")
    transl = motion_data.get("transl")
    root_rotations_mat = motion_data.get("root_rotations_mat")

    if rot6d is None or transl is None or keypoints3d is None:
        raise ValueError("Motion data must contain 'rot6d', 'transl', and 'keypoints3d'")

    # Helper to convert and remove batch dim
    def to_numpy(t):
        if isinstance(t, torch.Tensor):
            t = t.cpu().numpy()
        # Remove batch dim if present (take first sample)
        if len(t.shape) == 4:
            t = t[0]
        elif len(t.shape) == 3 and t.shape[0] == 1:
            t = t[0]
        return t

    return {
        "keypoints3d": to_numpy(keypoints3d),
        "rot6d": to_numpy(rot6d),
        "transl": to_numpy(transl),
        "root_rotations_mat": to_numpy(root_rotations_mat),
    }


def _export_with_retargeting(
    retarget_data: Dict[str, np.ndarray],
    output_path: Path,
    character_fbx: str,
    yaw_offset: float,
    scale: float,
) -> str:
    """Export motion with Mixamo retargeting."""
    from .hymotion.utils.retarget_fbx import retarget_fbx

    # Create temp file for NPZ data
    npz_path = tempfile.mktemp(suffix=".npz")
    np.savez(npz_path, **retarget_data)

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
        try:
            if os.path.exists(npz_path):
                os.unlink(npz_path)
        except PermissionError:
            pass  # Windows may hold the file briefly


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
