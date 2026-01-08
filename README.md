# HY-Motion Mixamo Exporter

A CLI tool that generates 3D character animations from text prompts using [HY-Motion 1.0](https://github.com/Tencent-Hunyuan/HY-Motion) and exports them as Mixamo-compatible FBX files.

## Features

- Generate realistic human motion from natural language descriptions
- Export animations as Mixamo-compatible FBX files
- Retarget animations to your own Mixamo characters
- Automatic setup of all dependencies on first run
- GPU VRAM auto-detection for optimal model selection
- BitsAndBytes quantization for lower VRAM usage

## Requirements

- **Python**: 3.10 or higher
- **GPU**: NVIDIA GPU with CUDA support
- **VRAM**: Minimum 8GB (for Lite model with int4 quantization)
- **OS**: Windows, Linux, or macOS

### Model Variants

| Model | Parameters | Description |
|-------|------------|-------------|
| `lite` | 0.46B | Faster, lower VRAM, good quality |
| `full` | 1B | Higher quality, requires more VRAM |

### VRAM Requirements

| Model | LLM Precision | Minimum VRAM |
|-------|---------------|--------------|
| Lite  | int4          | ~8GB         |
| Lite  | int8          | ~12GB        |
| Full  | int4          | ~12GB        |
| Full  | int8          | ~16GB        |
| Full  | none (fp16)   | ~24GB        |

Use `--model lite` or `--model full` to select. Default is `auto` (selects based on available VRAM).

## Installation

```bash
# Clone the repository
git clone https://github.com/your-username/hy-motion-mixamo-exporter.git
cd hy-motion-mixamo-exporter

# Install PyTorch with CUDA support first (if not already installed)
pip install torch --index-url https://download.pytorch.org/whl/cu124

# Install the package
pip install -e .
```

On **first run**, the tool will automatically:
1. Download HY-Motion model weights (~2-3GB for Lite)
2. Install the FBX SDK for Mixamo export
3. Download text encoder models (Qwen3-8B, CLIP) on first generation

This initial setup may take 5-10 minutes depending on your internet connection.

## Usage

### Getting a Mixamo Character (Required)

This tool requires a Mixamo character FBX file for exporting animations:

1. Go to [Mixamo](https://www.mixamo.com/#/?page=1&type=Character) (free, requires Adobe account)
2. Select any character
3. Download as **FBX** format (pose doesn't matter)
4. Use this file with the `-c` option

### Basic Usage

```bash
# Generate motion with your character
hy-motion-export "A person walks forward" -c character.fbx -o walking.fbx

# Different motion
hy-motion-export "A person jumps and waves" -c character.fbx -o jumping.fbx
```

### Choosing Model Variant

```bash
# Use Lite model (faster, ~8GB VRAM)
hy-motion-export "Walking forward" -c character.fbx --model lite -o walk.fbx

# Use Full model (better quality, ~12GB VRAM)
hy-motion-export "Walking forward" -c character.fbx --model full -o walk.fbx
```

### With Duration and Seed

```bash
# Set specific duration (0.5-12.0 seconds) and seed for reproducibility
hy-motion-export "Running fast" -c character.fbx -d 5.0 -s 42 -o running.fbx
```

### Full Options

```bash
hy-motion-export "Your motion description" \
    -c my_mixamo_char.fbx \
    -o result.fbx \
    -d 3.0 \
    --model lite \
    --precision int4 \
    -s 12345 \
    --cfg-scale 5.0
```

### Uninstall

To remove all downloaded models (~5-10GB):

```bash
# Interactive uninstall (asks for confirmation)
hy-motion-uninstall

# Skip confirmation
hy-motion-uninstall -y
```

## CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--character` | `-c` | Mixamo character FBX file for retargeting **(required)** |
| `--output` | `-o` | Output FBX file path (default: `output.fbx`) |
| `--duration` | `-d` | Motion duration in seconds, 0.5-12.0 (default: 3.0) |
| `--model` | `-m` | Model variant: `auto`, `full`, or `lite` (default: `auto`) |
| `--precision` | | LLM quantization: `auto`, `none`, `int8`, or `int4` (default: `auto`) |
| `--seed` | `-s` | Random seed for reproducibility |
| `--cfg-scale` | | Classifier-free guidance scale (default: 5.0) |
| `--reinstall` | | Force redownload models |

## Examples

```bash
# Walking
hy-motion-export "A person walks forward confidently" -c character.fbx -o walk.fbx

# Running
hy-motion-export "Running at full speed" -c character.fbx -d 3.0 -o run.fbx

# Jumping
hy-motion-export "Jump up and land softly" -c character.fbx -o jump.fbx

# Dancing
hy-motion-export "Dancing to upbeat music with arm movements" -c character.fbx -d 5.0 -o dance.fbx

# Sitting
hy-motion-export "Sitting down on the ground" -c character.fbx -o sit.fbx

# Waving
hy-motion-export "Standing and waving hello with right hand" -c character.fbx -o wave.fbx
```

## How It Works

1. **Text Encoding**: Your prompt is encoded using dual text encoders (CLIP + Qwen3-8B LLM) for rich semantic understanding
2. **Motion Generation**: HY-Motion's diffusion transformer generates SMPL-H motion data via flow matching
3. **Retargeting**: The SMPL-H motion is retargeted to your Mixamo character's skeleton
4. **FBX Export**: Final animation is exported as a standard FBX file

## Installation Directory

All data is stored in `~/.hy-motion-exporter/`:

```
~/.hy-motion-exporter/
├── models/               # Model weights
│   └── HY-Motion/
│       └── ckpts/
│           └── tencent/
│               ├── HY-Motion-1.0/
│               └── HY-Motion-1.0-Lite/
└── .installed            # Installation marker
```

Text encoder models (Qwen3-8B, CLIP) are cached in `~/.cache/huggingface/`.

## Troubleshooting

### "Insufficient GPU VRAM" Error

Try using a smaller model or more aggressive quantization:

```bash
hy-motion-export "walking" -c character.fbx --model lite --precision int4
```

### PyTorch Not Found or CUDA Not Available

Install PyTorch with CUDA support:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu124
```

### Reinstall Everything

If something goes wrong with the installation:

```bash
hy-motion-export "test" --reinstall
```

### FBX Export Issues

The tool requires the Autodesk FBX SDK. It's automatically installed from the PyNimation registry. If installation fails, you can install manually:

```bash
pip install fbxsdkpy --extra-index-url https://gitlab.inria.fr/api/v4/projects/18692/packages/pypi/simple
```

## Credits

- [HY-Motion 1.0](https://github.com/Tencent-Hunyuan/HY-Motion) by Tencent Hunyuan

## License

MIT License
