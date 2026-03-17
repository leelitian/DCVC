# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

DCVC-RT (Deep Contextual Video Compression - Real-Time) is a neural video codec from Microsoft (CVPR 2025). It achieves 100+ FPS for 1080p video coding with compression performance comparable to ECM.

### Core Features

- **Real-time Encoding**: 1080p video encoding/decoding at 125.2/112.8 FPS (NVIDIA A100)
- **4K Real-time Encoding**: Supports 4K video real-time encoding
- **Wide Bitrate Range**: Single model supports continuously adjustable bitrate control
- **Rate Control**: Supports rate control under dynamic network conditions
- **Unified YUV/RGB Encoding**: Supports both YUV420 and RGB content encoding

### Architecture

- **I-frames (Keyframes)**: Encoded using DMCI, can be decoded independently
- **P-frames (Predicted frames)**: Encoded using DMC, depends on previous frame
- **Latent Space**: Uses 8x8 pixel blocks (pixel shuffle/unshuffle), 192 channels (3 × 8 × 8)
- **Entropy Coding**: RANS (Range ANtonov Syedin) arithmetic coding with hyper-prior model

### Performance

| Resolution | Encoding | Decoding | Bitrate Savings vs VTM-17.0 | Bitrate Savings vs HM-16.25 |
|------------|----------|----------|------------------------------|----------------------------|
| 1080p      | 125.2 FPS | 112.8 FPS | 21% | 35% |
| 4K         | 31.5 FPS  | 27.5 FPS  | -    | -    |

Tested on UVG dataset with NVIDIA A100.

## Common Commands

### Environment Setup
```bash
conda create -n dcvc python=3.12
conda activate dcvc
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

### Building Extensions (Required Before Use)
```bash
sudo apt-get install cmake g++ ninja-build
cd ./src/cpp/ && pip install .
cd ../layers/extensions/inference/ && pip install .
```
If CUDA kernels fail to load, you'll see: `cannot import cuda implementation for inference, fallback to pytorch.`

### Download Pretrained Models

Download from [OneDrive](https://1drv.ms/f/c/2866592d5c55df8c/Esu0KJ-I2kxCjEP565ARx_YB88i0UnR6XnODqFcvZs4LcA?e=by8CO8) to `./checkpoints/`:
```
checkpoints/
├── cvpr2025_image.pth.tar  # Image model (I-frames)
└── cvpr2025_video.pth.tar  # Video model (P-frames)
```

### Running Tests
```bash
python test_video.py \
  --model_path_i ./checkpoints/cvpr2025_image.pth.tar \
  --model_path_p ./checkpoints/cvpr2025_video.pth.tar \
  --rate_num 4 \
  --test_config ./dataset_config_example_yuv420.json \
  --cuda 1 -w 1 --write_stream 1 \
  --force_zero_thres 0.12 \
  --output_path output.json \
  --force_intra_period -1 --reset_interval 64 \
  --force_frame_num -1 --check_existing 0 --verbose 0
```

Key options:
- `--rate_num`: Number of rate points (2-64)
- `--verbose`: 1 = sequence-level timing, 2 = frame-level timing
- `-w`: Number of GPUs (should match available GPUs)
- `--calc_ssim`: Calculate SSIM
- `--save_decoded_frame`: Save decoded frames

### Custom QP Testing
```bash
python test_video.py ... --qp_i 10 20 30 40 --qp_p 10 20 30 40
```

### Linting
The project uses `.flake8` with max line length 120.

## Code Architecture

### Directory Structure
```
DCVC/
├── test_video.py              # Main test entry script
├── src/
│   ├── models/                # Core compression models
│   │   ├── common_model.py    # Base CompressionModel class
│   │   ├── video_model.py     # DMC video encoder
│   │   ├── image_model.py     # DMCI image encoder
│   │   └── entropy_models.py  # Entropy coding models
│   ├── layers/                # Neural network layers and CUDA kernels
│   │   ├── layers.py          # Layer definitions
│   │   ├── cuda_inference.py  # CUDA inference kernels
│   │   └── extensions/inference/  # CUDA extensions
│   ├── utils/                 # Utilities
│   │   ├── video_reader.py    # Video reading (YUV420, PNG)
│   │   ├── video_writer.py   # Video writing
│   │   ├── metrics.py        # PSNR, MS-SSIM calculation
│   │   ├── transforms.py     # Color space conversion
│   │   ├── stream_helper.py  # Bitstream processing
│   │   └── common.py         # Common utilities
│   └── cpp/                   # C++ RANS arithmetic coding extension
└── DCVC-family/               # Other DCVC family models
```

### Core Models

| Component | Class | Purpose |
|-----------|-------|---------|
| Feature Extractor | FeatureExtractor | Extract current frame features |
| Encoder | Encoder | Encode features to latent space |
| Decoder | Decoder | Reconstruct features from latent space |
| Reconstruction | ReconGeneration | Reconstruct image from features |
| Hyper Encoder | HyperEncoder | Encode hyper-prior information |
| Hyper Decoder | HyperDecoder | Decode hyper-prior information |
| Prior Fusion | PriorFusion | Fuse multi-frame prior information |

### Key Configuration Parameters

```python
# Channel configuration (video_model.py)
g_ch_src_d = 192         # Source channels: 192 (3 × 8 × 8)
g_ch_recon = 320        # Reconstruction channels
g_ch_y = 128            # Latent space Y channels
g_ch_z = 192            # Hyper prior Z channels
g_ch_d = 256            # Feature channels

# QP configuration
qp_shift = [0, 8, 4]    # QP shift
extra_qp = 8            # Extra QP
```

### Core Innovations

- **Temporal Context Mining (TCM)**: Multi-scale temporal feature aggregation for better motion compensation
- **Pixel-Shuffle Downsampling**: Efficient spatial reduction using 8×8 pixel shuffle (192 channels)
- **Direct Motion Compression (DMC)**: Direct compression of motion information without explicit motion estimation
- **DMCI (Direct Mode with Conditional Integer)**: I-frame encoding with conditional integer factorization

## Dataset Configuration

### Supported Formats

1. **YUV420** - Common video test sequence format
2. **PNG** - RGB image sequences

### YUV420 Configuration

```json
{
  "root_path": "/media/data/",
  "test_classes": {
    "HEVC_B": {
      "test": 1,
      "base_path": "HEVC_B",
      "src_type": "yuv420",
      "sequences": {
        "BQTerrace_1920x1080_60.yuv": {
          "width": 1920,
          "height": 1080,
          "frames": 600,
          "intra_period": -1
        }
      }
    }
  }
}
```

### PNG (RGB) Configuration

```json
{
  "root_path": "/media/data/",
  "test_classes": {
    "RGB": {
      "test": 1,
      "base_path": "RGB",
      "src_type": "png",
      "sequences": {
        "seq1": {
          "width": 1920,
          "height": 1080,
          "frames": 96,
          "intra_period": 32
        }
      }
    }
  }
}
```

### Dataset Fields

| Field | Description |
|-------|-------------|
| `root_path` | Dataset root directory |
| `test_classes` | Test set classification |
| `test` | Enable test (0/1) |
| `base_path` | Path relative to root_path |
| `src_type` | Source type (yuv420 or png) |
| `sequences` | Video sequence list |
| `width` | Frame width |
| `height` | Frame height |
| `frames` | Total frame count |
| `intra_period` | I-frame period (-1 = only first frame is I-frame) |

### Common Test Datasets

| Dataset | Resolution | Frames | FPS |
|---------|------------|--------|-----|
| UVG (Beauty, Bosphorus, etc.) | 1920x1080 | 300-600 | 120 |
| MCL-JCV | 1920x1080 | 120-150 | - |
| HEVC_B | 1920x1080 | - | 24-60 |
| HEVC_E | 1280x720 | - | - |
| HEVC_C | 832x480 | - | - |
| HEVC_D | 416x240 | - | - |

### Preprocessing

- **Padding**: Input video is automatically padded to multiples of 64, cropped after reconstruction
- **Color Space Conversion**:
  - YUV420 → YUV444: `ycbcr420_to_444_np()`
  - RGB → YCbCr: `rgb2ycbcr()`
  - YCbCr → RGB: `ycbcr2rgb()`

### PSNR/SSIM Calculation

**YUV420 Content:**
```python
psnr = (6 * psnr_y + psnr_u + psnr_v) / 8
ssim = (6 * ssim_y + ssim_u + ssim_v) / 8
```

**RGB Content:**
```python
psnr = calc_psnr(rgb, rgb_rec)
msssim = calc_msssim_rgb(rgb, rgb_rec)
```

## Dependencies

- Python 3.12, CUDA 12.6, PyTorch 2.6
- C++/CUDA extensions built with pybind11 and torch.utils.cpp_extension

## CPU Performance Note

Arithmetic coding runs on CPU. For accurate speed measurements, ensure CPU runs at max frequency:
```bash
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Restore default
echo ondemand | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor
```

## Links

- [Paper (arXiv)](pdfs/DCVC-RT.pdf)
- [Project Page](https://dcvccodec.github.io/)
