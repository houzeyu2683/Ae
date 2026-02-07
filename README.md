# Passeriformes

A PyTorch implementation of autoencoder models for face image generation and reconstruction, built on the Diffusers library. The project provides two model variants: **Passeridae** (AutoencoderKL) and **Laniidae** (VQModel), covering both continuous and discrete latent space approaches.

## Project Structure

```
Passeriformes/
├── passeridae/              # AutoencoderKL model (continuous latent)
│   ├── _model_.py           # Model implementation
│   └── _framework_.py       # Training framework
├── laniidae/                # VQModel (discrete latent / vector-quantized)
│   ├── _model_.py           # Model implementation
│   └── _framework_.py       # Training framework
├── material/                # Data loading and preprocessing
│   ├── _hub_.py             # DataLoader hub (train/val/test)
│   └── storage/             # Dataset file lists and image data
├── visualization/           # TensorBoard integration
│   └── _dashboard_.py       # Metric logging
├── application/             # Model export and inference
│   ├── _luggage_.py         # ONNX export utility
│   └── _service_.py         # ONNX Runtime inference service
├── log/                     # Training logs and checkpoints
├── script-passeridae-*.py   # KL model pipeline scripts (1~4)
├── script-laniidae-*.py     # VQ model pipeline scripts (1~4)
└── environment.yaml         # Conda environment
```

## Model Architecture

### Passeridae (AutoencoderKL)

Based on `diffusers.AutoencoderKL`, uses continuous Gaussian latent space.

| Item | Value |
|------|-------|
| Encoder / Decoder blocks | 4 / 4 |
| Channel progression | 32, 64, 128, 256 |
| Latent channels | 32 |
| Loss function | MSE (pixel reconstruction) |

Key methods:
- `getCompression(image)` - Encode image to latent mean
- `getReconstruction(compression)` - Decode latent back to image
- `getCriteria(batch)` - Compute training loss

### Laniidae (VQModel)

Based on `diffusers.VQModel`, uses discrete codebook for quantized latent representations.

| Item | Value |
|------|-------|
| Encoder / Decoder blocks | 1 / 1 |
| Codebook size | 256 |
| Embedding dimension | 32 |
| Loss function | Commitment loss + MSE |

Key methods:
- `getRepresentation(image)` - Encode image to embedding, quantization, and token
- `getReconstruction(quantization)` - Decode quantized latent to image
- `getCriteria(batch)` - Compute training loss

## Installation

```bash
conda env create -f environment.yaml
conda activate Ae
```

### Core dependencies

- `torch` 2.9, `torchvision` 0.24
- `diffusers` 0.36
- `tensordict` 0.10
- `bitsandbytes` 0.49 (8-bit AdamW optimizer)
- `safetensors` 0.7
- `tensorboard` 2.20
- `onnxruntime-gpu` 1.23, `onnx` 1.20
- `pillow` 12.0

## Data Preparation

Place image paths in text files under `material/storage/`:

- `data.txt` - Training set (one relative path per line)
- `validation.txt` - Validation set
- `test.txt` - Test set (optional)

All paths are relative to `material/storage/`. Images are resized to 64x64 and normalized to [-1, 1].

## Pipeline Scripts

Each model has four pipeline scripts:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `script-{model}-1-fit-data.py` | Train the model |
| 2 | `script-{model}-2-average-weight.py` | Average multiple checkpoints |
| 3 | `script-{model}-3-exchange-module.py` | Export to ONNX format |
| 4 | `script-{model}-4-infer-data.py` | Run inference with ONNX Runtime |

Replace `{model}` with `passeridae` or `laniidae`.

### Training

```bash
# KL model (Passeridae)
python script-passeridae-1-fit-data.py

# VQ model (Laniidae)
python script-laniidae-1-fit-data.py
```

Training parameters are configured directly in the script:

```python
data = hub.getData(number=256)           # batch size
snapshot = 1000                          # checkpoint interval (steps)
total = -1                               # total steps (-1 = infinite)
accumulation = 4                         # gradient accumulation steps
history = './log/passeridae-2026-0208'   # output directory
```

Training uses mixed precision (`torch.amp.autocast`) and 8-bit AdamW optimizer with learning rate 1e-4.

### Resume training

Load a checkpoint before calling `fitWeight`:

```python
model.loadCheckpoint(path='./log/passeridae-2026-0117/checkpoint/210000.pt')
```

### Monitor training

```bash
tensorboard --logdir ./log/passeridae-2026-0208
```

Tracked metrics: `Loss/Data` (training loss) and `Loss/Validation` (validation loss), with sub-tags for each loss component.

### Export and inference

```bash
# Export model to ONNX
python script-passeridae-3-exchange-module.py

# Run inference using ONNX Runtime
python script-passeridae-4-infer-data.py
```

ONNX models are published to [GitHub Releases](https://github.com/houzeyu2683/Ae/releases) and downloaded automatically by `application.Service` at inference time.

## Output Structure

```
log/passeridae-2026-0208/
├── checkpoint/
│   ├── 1000.pt
│   ├── 2000.pt
│   └── ...
├── weight.pt                     # Averaged weights
└── events.out.tfevents.*         # TensorBoard logs
```

Checkpoints are saved in SafeTensors format.

## License

BSD 3-Clause License. See [LICENSE](LICENSE) for details.
