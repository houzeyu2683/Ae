# Ae - Autoencoder

A PyTorch implementation of Autoencoder (Ae) for face image generation and reconstruction, built on top of the Diffusers library's AutoencoderKL architecture.

## Overview

This project implements a VAE model for learning latent representations of face images. The model can:
- Encode face images into a lower-dimensional latent space
- Reconstruct images from their latent representations
- Generate new face images by sampling from the learned latent distribution

## Project Structure

```
Ae/
├── facility/           # Core model and training framework
│   ├── _sparrow_.py   # VAE model implementation (Sparrow)
│   └── _framework_.py # Training framework and utilities
├── material/          # Data loading and preprocessing
│   ├── _hub_.py      # DataLoader hub for train/val/test sets
│   └── storage/      # Dataset file lists (data.txt, validation.txt)
├── visualization/     # TensorBoard integration
│   └── _dashboard_.py # Logging metrics and images
├── script-fit-data.py    # Main training script
└── script-unit-test.py   # Data loading test script
```

## Model Architecture

**Sparrow** (facility/_sparrow_.py): The VAE model based on `diffusers.AutoencoderKL`
- Input/Output: 3-channel RGB images (64×64)
- Latent channels: 8
- Encoder/Decoder: 5-layer hierarchical structure with channels [32, 64, 128, 256, 256]
- Latent space: 8×4×4 = 128-dimensional

**Loss Function**:
- KL Divergence: Regularizes the latent distribution
- Mean Squared Error (MSE): Reconstruction quality
- Total Loss = scale × KL + MSE

## Installation

### Dependencies

```bash
pip install torch torchvision
pip install diffusers
pip install tensordict
pip install bitsandbytes
pip install safetensors
pip install tensorboard
pip install torchcodec
pip install tqdm
pip install pillow
```

## Data Preparation

1. Prepare your face image dataset
2. Create file lists in `material/storage/`:
   - `data.txt`: Training set image paths (one per line)
   - `validation.txt`: Validation set image paths
   - (Optional) `test.txt`: Test set image paths

**Note**: Image paths in the txt files should be relative to `material/storage/`.

## Usage

### Training

Edit [script-fit-data.py](script-fit-data.py) to configure training parameters:

```python
device = 'cuda'           # or 'cpu'
batch = 256              # training batch size
snapshot = 1000          # save checkpoint every N steps
total = -1               # total iterations (-1 for infinite)
accumulation = 1         # gradient accumulation steps
history = './exp/Dec31'  # experiment output directory
```

Run training:
```bash
python script-fit-data.py
```

**Key parameters**:
- `scale`: KL divergence weight in loss function (default: 1.0, adjustable in framework.fitWeight)
- `rate`: Learning rate (default: 1e-4)

### Resuming Training

Uncomment and modify the following line in [script-fit-data.py](script-fit-data.py):
```python
framework.loadWeight(path='./exp/Dec23-2/weight/400.pt')
```

### Monitoring Training

Launch TensorBoard to monitor training progress:
```bash
tensorboard --logdir ./exp/Dec31
```

Available visualizations:
- `Data/Loss(Total)`: Combined loss
- `Data/Loss(Divergence)`: KL divergence term
- `Data/Loss(Mean Squared Error)`: Reconstruction error
- `Validation`: Original vs. reconstructed images
- `Generation`: Randomly generated faces

### Testing Data Loading

Verify your dataset is set up correctly:
```bash
python script-unit-test.py
```

## Output Structure

Training outputs are saved to the specified `history` directory:
```
exp/Dec31/
├── weight/
│   ├── model.pt      # Full model (saved at start)
│   ├── 0.pt          # Checkpoint at step 0
│   ├── 1000.pt       # Checkpoint at step 1000
│   └── ...
└── events.out.tfevents.*  # TensorBoard logs
```

Checkpoints are saved using SafeTensors format (`.pt` files contain state_dict).

## Datasets

The `note` file contains links to potential face datasets:
- AFAD-Full
- Asian Regularization Images
- SCUT-FBP5500 V2 (Facial Beauty Rating)
- CASIA-Webface
- UMDFace
- VGG2
- Asian-celeb-112x112

## Key Classes

### Hub (material/_hub_.py)
Data management class for loading datasets.

**Methods**:
- `getData(batch)`: Get training DataLoader
- `getValidation(batch, reproducibility)`: Get validation DataLoader
- `getTest(batch, reproducibility)`: Get test DataLoader

### Sparrow (facility/_sparrow_.py)
VAE model implementation.

**Methods**:
- `activateLayer()`: Initialize the AutoencoderKL architecture
- `getDistribution(image)`: Encode image to latent distribution (mean, variance)
- `getReconstruction(reparameterization)`: Decode latent vector to image
- `getCriteria(batch, scale)`: Compute loss (KL + MSE)
- `getGeneration(number)`: Generate random faces by sampling latent space

### Framework (facility/_framework_.py)
Training orchestration and checkpoint management.

**Methods**:
- `fitWeight(data, snapshot, total, accumulation, validation)`: Main training loop
- `saveWeight(path)` / `loadWeight(path)`: Checkpoint I/O
- `saveModel(path)` / `loadModel(path)`: Full model I/O

### Dashboard (visualization/_dashboard_.py)
TensorBoard logging wrapper.

**Methods**:
- `insertElement(tag, value, number)`: Log scalar metrics
- `insertPicture(tag, image, number)`: Log image grids

## Advanced Configuration

### Modifying Model Architecture

Edit `activateLayer()` in [facility/_sparrow_.py](facility/_sparrow_.py):
```python
layer = diffusers.AutoencoderKL(
    in_channels=3,
    out_channels=3,
    latent_channels=8,
    block_out_channels=[32, 64, 128, 256, 256],  # Adjust channel sizes
    down_block_types=["DownEncoderBlock2D"] * 5,
    up_block_types=["UpDecoderBlock2D"] * 5
)
```

### Gradient Accumulation

For larger effective batch sizes on limited GPU memory:
```python
accumulation = 4  # Effective batch = batch * accumulation
```

### Mixed Precision Training

The framework uses `torch.amp.autocast` for automatic mixed precision training on CUDA devices.

## License

See [LICENSE](LICENSE) for details.