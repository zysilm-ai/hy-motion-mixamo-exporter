"""
Direct HY-Motion inference module.

This module provides a simple API for text-to-motion generation using HY-Motion,
bypassing ComfyUI entirely. It supports BitsAndBytes quantization for lower VRAM usage.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import torch
import yaml

from .config import MODELS_DIR


class HYMotionInference:
    """
    Direct HY-Motion inference wrapper with quantization support.

    This class loads the HY-Motion model and text encoder, providing a simple
    interface for generating motion from text prompts.
    """

    def __init__(
        self,
        model_name: str = "HY-Motion-1.0",
        quantization: str = "int4",
        device: str = "cuda",
        offload_to_cpu: bool = False,
    ):
        """
        Initialize HY-Motion inference.

        Args:
            model_name: Model name ("HY-Motion-1.0" or "HY-Motion-1.0-Lite")
            quantization: LLM quantization ("none", "int8", "int4")
            device: Device to use ("cuda" or "cpu")
            offload_to_cpu: Whether to offload LLM to CPU after encoding
        """
        self.model_name = model_name
        self.quantization = quantization
        self.device = device
        self.offload_to_cpu = offload_to_cpu

        self.pipeline = None
        self.text_encoder = None
        self._loaded = False

        # Find model path
        self.model_path = self._find_model_path()

    def _find_model_path(self) -> Path:
        """Find the model checkpoint directory."""
        # Check in MODELS_DIR (standard location)
        model_dir = MODELS_DIR / "ckpts" / "tencent" / self.model_name
        if model_dir.exists():
            return model_dir

        # Check in HuggingFace cache
        hf_cache = Path.home() / ".cache" / "huggingface" / "hub"
        for path in hf_cache.glob(f"models--tencent--HY-Motion*"):
            snapshot_dir = path / "snapshots"
            if snapshot_dir.exists():
                for snapshot in snapshot_dir.iterdir():
                    if (snapshot / self.model_name / "config.yml").exists():
                        return snapshot / self.model_name

        raise FileNotFoundError(
            f"Model {self.model_name} not found. Please run setup first."
        )

    def load(self) -> None:
        """Load the model and text encoder."""
        if self._loaded:
            return

        print(f"[HY-Motion] Loading model from {self.model_path}")

        # Load config
        config_path = self.model_path / "config.yml"
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Import pipeline and network loaders
        from .hymotion.utils.loaders import load_object

        # Build pipeline
        print("[HY-Motion] Building pipeline...")
        self.pipeline = load_object(
            config["train_pipeline"],
            config["train_pipeline_args"],
            network_module=config["network_module"],
            network_module_args=config["network_module_args"],
        )

        # Load checkpoint
        ckpt_path = self.model_path / "latest.ckpt"
        print(f"[HY-Motion] Loading checkpoint: {ckpt_path}")
        checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = checkpoint["model_state_dict"]

        # Handle mean/std buffers
        if "mean" in state_dict and "std" in state_dict:
            self.pipeline.register_buffer("mean", state_dict["mean"].clone())
            self.pipeline.register_buffer("std", state_dict["std"].clone())

        self.pipeline.load_state_dict(state_dict, strict=False)
        self.pipeline.to(self.device)
        self.pipeline.eval()

        print("[HY-Motion] Pipeline loaded")

        # Load text encoder with quantization
        print(f"[HY-Motion] Loading text encoder with {self.quantization} quantization...")
        self._load_text_encoder(config)

        self._loaded = True
        print("[HY-Motion] Model ready!")

    def _load_text_encoder(self, config: dict) -> None:
        """Load the text encoder with quantization support."""
        from .hymotion.network.text_encoders.text_encoder import HYTextModel

        text_encoder_cfg = config["train_pipeline_args"].get("text_encoder_cfg", {})

        # Create text encoder with quantization
        self.text_encoder = HYTextModel(
            llm_type=text_encoder_cfg.get("llm_type", "qwen3"),
            max_length_llm=text_encoder_cfg.get("max_length_llm", 128),
            sentence_emb_type="clipl",
            quantization=self.quantization if self.quantization != "none" else None,
        )

        # Move to device (quantized models handle device mapping automatically)
        if self.quantization == "none":
            self.text_encoder.to(self.device)

        self.text_encoder.eval()

        # Store reference in pipeline for text encoding
        self.pipeline.text_encoder = self.text_encoder

    def encode_text(self, text: str) -> Dict[str, torch.Tensor]:
        """
        Encode text prompt to conditioning tensors.

        Args:
            text: Text prompt describing the motion

        Returns:
            Dictionary with encoded text features
        """
        self.load()

        text_list = [text] if isinstance(text, str) else text

        with torch.no_grad():
            vtxt_raw, ctxt_raw, ctxt_length = self.text_encoder.encode(text_list)

        return {
            "text_vec_raw": vtxt_raw,
            "text_ctxt_raw": ctxt_raw,
            "text_ctxt_raw_length": ctxt_length,
        }

    @torch.no_grad()
    def generate(
        self,
        prompt: str,
        duration: float = 3.0,
        seed: Optional[int] = None,
        cfg_scale: float = 5.0,
        progress_callback: Optional[callable] = None,
    ) -> Dict[str, Any]:
        """
        Generate motion from text prompt.

        Args:
            prompt: Text description of the motion
            duration: Duration in seconds (0.5 to 12.0)
            seed: Random seed for reproducibility
            cfg_scale: Classifier-free guidance scale
            progress_callback: Optional callback(step, total) for progress

        Returns:
            Dictionary containing:
                - rot6d: Rotation data in 6D format (N, 22, 6)
                - transl: Translation data (N, 3)
                - vertices3d: Mesh vertices (N, V, 3)
                - keypoints3d: Joint positions (N, J, 3)
                - smpl_data: SMPL-H format data for export
                - text: Original prompt
        """
        self.load()

        # Validate duration
        duration = max(0.5, min(12.0, duration))

        # Generate seed if not provided
        if seed is None:
            seed = torch.randint(0, 2**31, (1,)).item()

        print(f"[HY-Motion] Generating motion...")
        print(f"  Prompt: {prompt}")
        print(f"  Duration: {duration}s")
        print(f"  Seed: {seed}")
        print(f"  CFG Scale: {cfg_scale}")

        # Encode text
        hidden_state_dict = self.encode_text(prompt)

        # Generate motion
        output = self.pipeline.generate(
            text=prompt,
            seed_input=[seed],
            duration_slider=duration,
            cfg_scale=cfg_scale,
            hidden_state_dict=hidden_state_dict,
            progress_callback=progress_callback,
        )

        print("[HY-Motion] Motion generated successfully!")

        return output

    def generate_multiple(
        self,
        prompt: str,
        duration: float = 3.0,
        seeds: List[int] = None,
        num_samples: int = 1,
        cfg_scale: float = 5.0,
    ) -> Dict[str, Any]:
        """
        Generate multiple motion variations from the same prompt.

        Args:
            prompt: Text description of the motion
            duration: Duration in seconds
            seeds: List of seeds (auto-generated if None)
            num_samples: Number of samples if seeds not provided
            cfg_scale: Classifier-free guidance scale

        Returns:
            Dictionary with batched outputs
        """
        self.load()

        if seeds is None:
            seeds = [torch.randint(0, 2**31, (1,)).item() for _ in range(num_samples)]

        # Encode text
        hidden_state_dict = self.encode_text(prompt)

        # Generate motion
        output = self.pipeline.generate(
            text=prompt,
            seed_input=seeds,
            duration_slider=duration,
            cfg_scale=cfg_scale,
            hidden_state_dict=hidden_state_dict,
        )

        return output

    def unload(self) -> None:
        """Unload models and free VRAM."""
        if self.pipeline is not None:
            del self.pipeline
            self.pipeline = None

        if self.text_encoder is not None:
            del self.text_encoder
            self.text_encoder = None

        self._loaded = False

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print("[HY-Motion] Models unloaded, VRAM freed")
