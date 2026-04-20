# Self-Distilled Diffusion Experiment Suite

This is a notebook-first research template for testing **Self-Distilled Diffusion (SDD)** on **CIFAR-10** and **Tiny ImageNet (64x64)** with a lightweight **UNet-based diffusion model**.

It is tuned for a single-GPU workstation such as an **RTX 4070**, with:
- mixed precision training
- gradient accumulation
- EMA teacher updates
- `tqdm` progress bars
- Weights & Biases logging
- ablation toggles for **centering**, **sharpening**, and **gating**

## Environment

The project pins PyTorch to the current stable 2.7.1 release line and matching torchvision/torchaudio wheels. Official PyTorch install pages list stable PyTorch 2.7.0/2.7.x wheels and the 2.7.1 GA announcement confirms 2.7.1 availability; torchvision 0.22.1 wheels are published on the official PyTorch wheel index. citeturn751335search0turn751335search5turn550293search3

Recommended install:
```bash
pip install -r requirements.txt
```

## Datasets

- **CIFAR-10**: fast ablation runs and sanity checks
- **Tiny ImageNet (64x64)**: stronger representation and generation experiments

The notebooks include dataset download and preparation cells for both datasets.

## Notebook workflow

The project is notebook-first. All experiment and visualization workflows live in `.ipynb` files:

| Notebook | Description / Experiment | Key purpose |
|---|---|---|
| `01_prepare_data.ipynb` | Data preparation | Download and preprocess CIFAR-10 / Tiny ImageNet |
| `02_train_baseline_cifar10.ipynb` | Baseline training | Train vanilla diffusion model on CIFAR-10 |
| `03_train_sdd_cifar10.ipynb` | SDD training (CIFAR-10) | Apply Self-Distilled Diffusion on CIFAR-10 |
| `04_train_sdd_tiny_imagenet.ipynb` | SDD training (Tiny ImageNet) | Scale SDD to 64x64 dataset |
| `05_ablation_sweeps.ipynb` | Ablation experiments | Toggle centering, sharpening, gating, etc. |
| `06_eval_fid.ipynb` | FID evaluation | Quantitative generation quality |
| `07_eval_linear_probe.ipynb` | Linear probe | Representation quality evaluation |
| `08_visualize_features.ipynb` | Feature visualization | Inspect learned feature maps |
| `09_compare_runs.ipynb` | Run comparison | Compare multiple experiment outputs |
| `10_timestep_analysis.ipynb` | Timestep behavior | Analyze diffusion timestep effects |
| `11_feature_layer_ablation.ipynb` | Feature-layer ablation | Which UNet layer is the best distillation target? |
| `12_training_curves.ipynb` | Training curves | Does SDD converge faster and improve FID vs baseline? |
| `13_gating_analysis.ipynb` | Gating distribution | Which timesteps actually receive distillation signal? |
| `14_ema_momentum_sweep.ipynb` | EMA momentum sweep | How sensitive is SDD to teacher staleness? |
| `15_sample_grid.ipynb` | Sample comparison grid | Qualitative visual difference between baseline and SDD |
| `16_umap_tsne_features.ipynb` | UMAP / t-SNE visualization | Does SDD produce more class-separable representations? |

## Key ablations

The configs support:
- full SDD
- `w/o Centering`
- `w/o Sharpening`
- `w/o Gating`
- `w/o Projection Head`
- `MSE only`
- `EMA only`
- soft vs hard timestep gating
- temperature sweeps
- gating range sweeps
- EMA momentum sweeps
- distillation weight sweeps
- feature layer sweeps

## Project layout

```text
sdd-diffusion/
├── configs/
├── notebooks/
├── src/
├── scripts/
├── outputs/
└── data/
```

## How to run

Open the notebooks in order, starting from data preparation. Each training notebook exposes a config cell at the top so you can switch datasets or ablations without editing the source package.

## Notes

- The UNet is intentionally lightweight and RTX 4070-friendly.
- The trainer uses `tqdm` for live progress visualization.
- The wandb logger is optional and controlled through config, but enabled by default in the notebooks.
- Tiny ImageNet validation preparation is included as a notebook step because the raw dataset ships in a non-ImageFolder format.

### New API functions

All new experiment functions live in `src/experiments/notebook_api.py` and are exported from `src/experiments/__init__.py`:

```python
from src.experiments import (
    # Experiment 1 — feature layer
    run_feature_layer_ablation, FEATURE_LAYERS,

    # Experiment 2 — training curves
    train_with_curves, compare_training_curves,

    # Experiment 3 — gating analysis
    collect_gate_histogram, run_gating_analysis,

    # Experiment 4 — EMA sweep
    run_ema_momentum_sweep,

    # Experiment 5 — sample grids
    generate_sample_grid, generate_comparison_grid,

    # Experiment 6 — UMAP / t-SNE
    run_umap_comparison, run_tsne_comparison,
)
```

### Multi-layer feature extraction

`UNetModel.forward()` now accepts a `feature_layer` argument:

```python
out, feat = model(x, t, return_features=True, feature_layer="skip1")
# feature_layer options: "bottleneck" (default), "skip1", "skip2", "decoder1"
```

### Running all tests

```bash
pytest tests/ -v
```

The test suite now covers UNet forward/feature extraction, diffusion schedules,
gating (hard/soft/none), DINO distillation loss, EMA update correctness, and
the total loss function including per-sample gating behaviour.
