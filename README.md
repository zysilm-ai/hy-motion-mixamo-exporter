# HY-Motion Mixamo Exporter

A CLI tool that generates 3D character animations from text prompts using [HY-Motion 1.0](https://github.com/Tencent-Hunyuan/HY-Motion) and exports them as Mixamo-compatible FBX files.

## Features

- Generate realistic human motion from natural language descriptions
- Export animations as Mixamo-compatible FBX files
- Retarget animations to your own Mixamo characters
- Automatic setup of all dependencies on first run
- GPU VRAM auto-detection for optimal model selection

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

# Install the package
pip install -e .
```

On **first run**, the tool will automatically:
1. Install ComfyUI and required plugins
2. Download HY-Motion model weights (~2-3GB for Lite)
3. Install the FBX SDK for Mixamo export
4. Download text encoder models (Qwen3-8B, CLIP)

This initial setup may take 10-15 minutes depending on your internet connection.

## Usage

### Basic Usage

```bash
# Generate a simple walking animation
hy-motion-export "A person walks forward"

# Specify output file
hy-motion-export "A person jumps and waves" -o jumping.fbx
```

### With Custom Mixamo Character

```bash
# Use your own Mixamo character for retargeting
hy-motion-export "Dancing happily" -c my_character.fbx -o dance.fbx
```

### With Duration and Seed

```bash
# Set specific duration (in seconds) and seed for reproducibility
hy-motion-export "Running fast" -d 5.0 -s 42 -o running.fbx
```

### Full Options

```bash
hy-motion-export "Your motion description" \
    --output result.fbx \
    --character my_mixamo_char.fbx \
    --duration 3.0 \
    --model lite \
    --precision int4 \
    --seed 12345 \
    --keep-server
```

## CLI Options

| Option | Short | Description |
|--------|-------|-------------|
| `--output` | `-o` | Output FBX file path (default: `output.fbx`) |
| `--character` | `-c` | Custom Mixamo character FBX for retargeting |
| `--duration` | `-d` | Motion duration in seconds (auto-detected if not set) |
| `--model` | `-m` | Model variant: `auto`, `full`, or `lite` (default: `auto`) |
| `--precision` | | LLM quantization: `auto`, `none`, `int8`, or `int4` (default: `auto`) |
| `--seed` | `-s` | Random seed for reproducibility |
| `--port` | `-p` | ComfyUI server port (default: 8188) |
| `--keep-server` | | Keep ComfyUI server running after generation |
| `--reinstall` | | Force reinstall ComfyUI and plugins |

## Examples

### Character Animations

```bash
# Walking
hy-motion-export "A person walks forward confidently" -o walk.fbx

# Running
hy-motion-export "Running at full speed" -d 3.0 -o run.fbx

# Jumping
hy-motion-export "Jump up and land softly" -o jump.fbx

# Dancing
hy-motion-export "Dancing to upbeat music with arm movements" -d 5.0 -o dance.fbx

# Sitting
hy-motion-export "Sitting down on the ground" -c character.fbx -o sit.fbx

# Waving
hy-motion-export "Standing and waving hello with right hand" -o wave.fbx
```

### Combining with Custom Characters

Download a character from [Mixamo](https://www.mixamo.com/) (FBX format, T-pose, no animation), then:

```bash
hy-motion-export "Walking while looking around" -c mixamo_character.fbx -o custom_walk.fbx
```

## How It Works

1. **Text Encoding**: Your prompt is encoded using dual text encoders (CLIP + Qwen3-8B LLM) for rich semantic understanding
2. **Motion Generation**: HY-Motion's diffusion model generates SMPL-H motion data
3. **Retargeting**: The SMPL-H motion is retargeted to Mixamo's 52-joint skeleton
4. **FBX Export**: Final animation is exported as a standard FBX file

## Installation Directory

All data is stored in `~/.hy-motion-exporter/`:

```
~/.hy-motion-exporter/
├── comfyui/              # ComfyUI installation
│   ├── models/           # Model weights
│   ├── input/            # Input files (custom characters)
│   └── output/           # Generated FBX files
└── .installed            # Installation marker
```

## Troubleshooting

### "Insufficient GPU VRAM" Error

Try using a smaller model or more aggressive quantization:

```bash
hy-motion-export "walking" --model lite --precision int4
```

### Server Already Running

If you see port conflicts, either:
- Use `--keep-server` on previous runs to reuse the server
- Kill the existing process and try again
- Use a different port with `--port 8189`

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
- [ComfyUI-HY-Motion1](https://github.com/jtydhr88/ComfyUI-HY-Motion1) by jtydhr88
- [ComfyUI](https://github.com/comfyanonymous/ComfyUI)

## License

MIT License
